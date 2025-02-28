#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

#include <vector>
#include <algorithm>
#include <string>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::interpreter;

static constexpr int QKRot_D = 64, QVO_D = 512, QVO_Dd2 = QVO_D/2, NUM_ROWS = 32, PAGE_SIZE = 256, ITERS_PER_PAGE = PAGE_SIZE / NUM_ROWS;
using qrot_tile           = st_bf<64, QKRot_D>;
using qvo_tile            = st_bf<64, QVO_D>;
using q_global            = kittens::gl<bf16, -1, -1, -1, QKRot_D, qrot_tile>; // B * R * H * D_QKRot_D
using qv_global           = kittens::gl<bf16, -1, -1, -1, QVO_D, qvo_tile>; // B * R * H * D_QVO_D
using kcache_tile         = st_bf<NUM_ROWS, QKRot_D>;
using vcache_tile         = st_bf<NUM_ROWS, QVO_D>; // we need the v_tile for later
using vcache_tile2        = st_bf<NUM_ROWS, QVO_Dd2>; // we need the v_tile for later
using kcache_global       = kittens::gl<bf16, 1, -1, PAGE_SIZE, QKRot_D, kcache_tile>; // 1 * #page * pagesize * QKRot_D
using vcache_global       = kittens::gl<bf16, 1, -1, PAGE_SIZE, QVO_D, vcache_tile>; // 1 * #page * pagesize * QVO_D
using ops_global          = kittens::gl<bf16, 1, -1, -1, 8>;
using instructions_global = kittens::gl<int, 1, -1, -1, 32>;
using table_global        = kittens::gl<int, 1, 1, -1, -1>; // B * (max # pages)
using o_tile              = st_bf<64, QVO_D>;
using o_tile_fl           = st_fl<16, QVO_D>;
using o_global            = kittens::gl<bf16, -1, -1, -1, QVO_D, st_bf<16, QVO_Dd2>, st_bf<16, QVO_D/8>>; // B * NEWTOKENS * H * D_VO

using o_scratch_global    = kittens::gl<float, -1, -1, 16, QVO_D, st_fl<16, QVO_D/8>, st_fl<16,256>>; // For partial O's
using lvec_scratch_global = kittens::gl<float,  1, -1, -1, 16, sv_fl<16>>; // For partial O's
using semaphore_global    = kittens::gl<int,    1,  1,  -1, -1>;            // 1 * 1 * uid * NEWTOKENS

struct config {
    struct globals {
        using instructions_global = instructions_global;
        instructions_global instructions;
        q_global Q;
        qv_global QV;
        kcache_global K_cache;
        vcache_global V_cache;
        table_global Table;
        o_global O;
        o_scratch_global O_scratch;
        lvec_scratch_global Lvec_scratch;
        semaphore_global semaphore;
        const float Softmax_scale;
        int tic;
        gl<int, 1, -1, -1, 2> timings;
        int dynamic_shared_memory() { return 226000; }
        dim3 grid()  { return dim3(132); } //dim3(Q.batch * ((Q.depth + 3) / 4)); }
        dim3 block() { return dim3((8+4)*WARP_THREADS); }
    };
};

struct location {
    int batch_idx; // batch_idx >=0, otherwise it's the negative index, minus one, into scratch
    int seq_idx;
};
struct partial_layout {
    using globals = config::globals;
    struct input_block { kcache_tile kcache; vcache_tile vcache; };
    struct scratch_block { qrot_tile qrot; qvo_tile qvo; st_bf<64, kcache_tile::rows> att_block; sv_fl<64> max_vec, norm_vec; };
    struct finish_block { st_fl<16, QVO_Dd2> o[4][2]; sv_fl<16> lvec[4]; };
    struct common_state {
        int uid;
        location dst;
        int q_batch_idx;
        int q_seq_idx;
        int start_pos; // MUST BE A MULTIPLE OF PAGE_SIZE
        int end_pos; // One past the last position to load
        int length; // the length of the overall sequence in question
    };
    struct consumer_state {
        col_vec<rt_fl<16, kcache_tile::rows>> max_vec, norm_vec;
        rt_fl<16, QVO_Dd2> o;
    };
};
struct partial_template {
    using config = config;
    using layout = partial_layout;
    static constexpr int opcode = 1;
    static constexpr int INPUT_PIPE_STAGES = 2;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        args.common.uid         =  args.instruction[1];
        args.common.dst         = {args.instruction[2],
                                   args.instruction[3]};
        args.common.q_batch_idx =  args.instruction[4];
        args.common.q_seq_idx   =  args.instruction[5];
        args.common.start_pos   =  args.instruction[6];
        args.common.end_pos     =  args.instruction[7];
        args.common.length      =  args.instruction[8];
        args.num_iters          = (args.common.end_pos - args.common.start_pos + NUM_ROWS - 1) / NUM_ROWS;
        args.common.length    -= (args.globals.Q.depth() - (args.common.q_seq_idx + warpgroup::warpid()) - 1); // adjust for the causal mask
    }
    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {}
        __device__ static inline void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == 0) {
                int global_load_idx = args.iter % ITERS_PER_PAGE;
                int next_page_id = args.globals.Table[coord<>{args.common.q_batch_idx, (args.common.start_pos/PAGE_SIZE) + (args.iter/ITERS_PER_PAGE)}];
                // next page we need to load?
                // tma::expect(args.inputs_arrived, args.input.kcache, args.input.vcache);
                tma::expect(args.inputs_arrived, args.input.vcache);
                // cache shape is 1, # pages, page size, QK_D
                // tma::load_async(args.input.kcache, args.globals.K_cache, {0, next_page_id, global_load_idx, 0}, args.inputs_arrived);
                tma::load_async(args.input.vcache, args.globals.V_cache, {0, next_page_id, global_load_idx, 0}, args.inputs_arrived);
            }
            else if(laneid() == 0) arrive(args.inputs_arrived);
            warpgroup::sync(5);
        }
    };
    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
            args.timing.start = clock64();
            zero(args.state.norm_vec);
            neg_infty(args.state.max_vec);
            zero(args.state.o);
            // if (warpgroup::groupid() == 0) {
            //     auto qrot_st = subtile_inplace<16, QKRot_D>(args.scratch.qrot, {warpgroup::warpid(), warpgroup::groupid()});
            //     load_async(qrot_st, args.globals.Q, {args.common.q_batch_idx, args.common.q_seq_idx + warpgroup::warpid(), 0, warpgroup::groupid()});
            // }
            auto qvo_st = subtile_inplace<16, QVO_Dd2>(args.scratch.qvo, {warpgroup::warpid(), warpgroup::groupid()});
            load_async(qvo_st, args.globals.QV, {args.common.q_batch_idx, args.common.q_seq_idx + warpgroup::warpid(), 0, warpgroup::groupid()});
            load_async_wait();
            group<8>::sync(10);
        }
        template<bool do_right_fill> __device__ static inline void internal_compute(consumer_compute_args<layout> args) {
            // 1.44269504089f is from exp2
            const float SOFTMAX_TEMPERATURE = args.globals.Softmax_scale * 1.44269504089f;

            col_vec<rt_fl<16, kcache_tile::rows>> local_max_vec, local_norm_vec;
            col_vec<rt_fl<16, kcache_tile::rows>> max_vec_last_scaled, max_vec_scaled;

            copy(local_max_vec,  args.state.max_vec);
            copy(local_norm_vec, args.state.norm_vec);

            if(warpgroup::groupid() == 0) {
                // A = Q @ K.T
                rt_fl<16, kcache_tile::rows> att_block_fp32;
                // warpgroup::mm_ABt(att_block_fp32, args.scratch.qrot, args.input.kcache);
                warpgroup::mm_ABt(att_block_fp32, args.scratch.qvo, args.input.vcache);

                mul(max_vec_last_scaled, local_max_vec, SOFTMAX_TEMPERATURE);

                warpgroup::mma_async_wait();
                // softmax
                if constexpr (do_right_fill) { // need to mask out a bunch of entries in the last page
                    const int length = args.common.length - args.common.start_pos - args.iter*NUM_ROWS;
                    right_fill(att_block_fp32, att_block_fp32, length, -9999999999.f);
                }

                row_max(local_max_vec, att_block_fp32, local_max_vec);
                mul(max_vec_scaled, local_max_vec, SOFTMAX_TEMPERATURE);

                mul(att_block_fp32, att_block_fp32, SOFTMAX_TEMPERATURE);
                sub_row(att_block_fp32, att_block_fp32, max_vec_scaled);
                
                exp2(att_block_fp32, att_block_fp32);
                
                sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
                exp2(max_vec_last_scaled, max_vec_last_scaled);
                warpgroup::store(args.scratch.max_vec, max_vec_last_scaled);
                
                mul(local_norm_vec, local_norm_vec, max_vec_last_scaled);
                row_sum(local_norm_vec, att_block_fp32, local_norm_vec);
                warpgroup::store(args.scratch.att_block, att_block_fp32);
            }
            group<8>::sync(10);

            warpgroup::load(max_vec_last_scaled, args.scratch.max_vec);
            mul_row(args.state.o, args.state.o, max_vec_last_scaled); // normalize o_reg before mma

            // O += A @ V
            auto (&v_smem)[2] = reinterpret_cast<vcache_tile2(&)[2]>(args.input.vcache);
            warpgroup::mma_AB(args.state.o, args.scratch.att_block, v_smem[warpgroup::groupid()]);

            copy(args.state.max_vec, local_max_vec);
            copy(args.state.norm_vec, local_norm_vec);

            warpgroup::mma_async_wait();
            if(warpgroup::laneid() == 0) arrive(args.inputs_finished, WARPGROUP_WARPS); // done!
        }
        __device__ static inline void compute(consumer_compute_args<layout> args) {
            if(args.iter >= args.num_iters-2) internal_compute<true>(args);
            else internal_compute<false>(args);
        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
            col_vec<rt_fl<16, kcache_tile::rows>> local_max_vec, local_norm_vec;

            copy(local_norm_vec, args.state.norm_vec);
            copy(local_max_vec, args.state.max_vec);

            if (warpgroup::groupid() == 0) warpgroup::store(args.scratch.norm_vec, local_norm_vec);
            group<8>::sync(10);
            if(warpgroup::groupid() == 1) warpgroup::load(local_norm_vec, args.scratch.norm_vec);
            div_row(args.state.o, args.state.o, local_norm_vec);

            if(args.common.dst.batch_idx >= 0) { // batch is meaningful
                auto &o_smem = reinterpret_cast<st_bf<16, QVO_Dd2>&>(args.finish.o[warpgroup::warpid()][warpgroup::groupid()]);
                store(o_smem, args.state.o);
                __syncwarp();
                tma::store_async(args.globals.O, o_smem, {args.common.dst.batch_idx, args.common.dst.seq_idx+warpgroup::warpid(), 0, warpgroup::groupid()});
            }
            else { // write out directly to O scratch, without going through smem
                if(warpgroup::groupid() == 0) {
                    mul(local_max_vec, local_max_vec, args.globals.Softmax_scale * 1.44269504089f);
                    log2(local_norm_vec, local_norm_vec);
                    add(local_norm_vec, local_norm_vec, local_max_vec); // l_vec = log2(norm_vec) + max_vec
                    store(args.finish.lvec[warpgroup::warpid()], local_norm_vec);
                    __syncwarp();
                    tma::store_async(args.globals.Lvec_scratch, args.finish.lvec[warpgroup::warpid()], {-args.common.dst.batch_idx-1, args.common.dst.seq_idx+warpgroup::warpid(), 0});
                }

                store(args.finish.o[warpgroup::warpid()][warpgroup::groupid()], args.state.o);
                __syncwarp();
                tma::store_async(args.globals.O_scratch, args.finish.o[warpgroup::warpid()][warpgroup::groupid()], {-args.common.dst.batch_idx-1, args.common.dst.seq_idx+warpgroup::warpid(), 0, warpgroup::groupid()});
            }
            tma::store_async_wait(); // not just read wait
            group<8>::sync(10);
            if(args.common.dst.batch_idx < 0) {
                if(group<8>::laneid() < 4 && args.common.dst.seq_idx + group<8>::laneid() < args.globals.O_scratch.depth()) {
                    args.globals.semaphore[{-args.common.dst.batch_idx-1, args.common.dst.seq_idx + group<8>::laneid()}] = args.globals.tic;
                }
            }
            if(warpgroup::laneid() == 0) arrive(args.finish_finished, WARPGROUP_WARPS); // done!
            args.timing.end = clock64();
        }
    };
};
struct reduction_layout {
    using globals = config::globals;
    struct input_block   { st_fl<16, QVO_D/8> o[8]; sv_fl<16> lvec; sv_fl<16> padding[15]; };
    struct scratch_block { st_fl<16, QVO_D/8> o[8]; sv_fl<16> lvec; }; // used both for setup load and finish store
    struct common_state {
        int uid;
        // int num_iters; // same as the number of active load_uid's, marked here for instruction clarity but we just use args.num_iters instead.
        location dst; // again, negative batch means we're writing to O scratch, seq_idx is consistent
        int src_uid;
    };
    struct producer_state {
        int load_uid[26];
    };
    struct consumer_state {
        rt_fl<16, QVO_D/8> o;
        col_vec<rt_fl<16, kcache_tile::rows>> lvec;
    };
};
struct reduction_template {
    using config = config;
    using layout = reduction_layout;
    static constexpr int opcode = 2;
    static constexpr int INPUT_PIPE_STAGES = 4;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        args.common.uid     =  args.instruction[1];
        args.num_iters      =  args.instruction[2];
        args.common.dst     = {args.instruction[3],
                               args.instruction[4]};
        args.common.src_uid =  args.instruction[5];
    }
    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {
            #pragma unroll
            for(int i = 0; i < 26; i++) {
                args.state.load_uid[i] = args.instruction[6+i];
            }
        }
        __device__ static inline void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == args.iter%4) {
                // spinloop until we're ready
                int load_uid = args.state.load_uid[args.iter];
                if(laneid() == 0) while(*(volatile int*)&args.globals.semaphore[{load_uid, args.common.dst.seq_idx}] != args.globals.tic) {}
                __syncwarp();
                // next page we need to load?
                tma::expect(args.inputs_arrived, args.input.o, args.input.lvec);
                #pragma unroll
                for(int i = 0; i < 8; i++) {
                    tma::load_async(args.input.o[i], args.globals.O_scratch, {load_uid, args.common.dst.seq_idx, 0, i}, args.inputs_arrived);
                }
                tma::load_async(args.input.lvec, args.globals.Lvec_scratch, {load_uid, args.common.dst.seq_idx, 0}, args.inputs_arrived);
                if(laneid() == 0) arrive(args.inputs_arrived, 3);
            }
        }
    };
    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
            // If we are doing a reduction, we need to spinloop until we have confirmation that all the partial results have been written out.
            if(threadIdx.x == 0) { // easier to have a single thread spin
                while(*(volatile int*)&args.globals.semaphore[{args.common.src_uid, args.common.dst.seq_idx}] != args.globals.tic) {} // note volatile, L1 is not guaranteed to be coherent.
            }
            group<8>::sync(11); // all warps must sync here.
            args.timing.start = clock64(); // Now the instruction really starts.
            load_async(args.scratch.o[group<8>::warpid()], args.globals.O_scratch, {args.common.src_uid, args.common.dst.seq_idx, 0, group<8>::warpid()});
            if(warpid() == 0) {
                load_async(args.scratch.lvec, args.globals.Lvec_scratch, {args.common.src_uid, args.common.dst.seq_idx, 0});
            }
            load_async_wait();
            __syncwarp();
            load(args.state.o, args.scratch.o[group<8>::warpid()]);
            group<8>::sync(11); // all warps must sync here.
            load(args.state.lvec, args.scratch.lvec);
        }
        __device__ static inline void compute(consumer_compute_args<layout> args) {
            col_vec<rt_fl<16, kcache_tile::rows>> lvec, max_lvec, sum_lvec;
            rt_fl<16, QVO_D / 8> o;
            load(o, args.input.o[group<8>::warpid()]);
            load(lvec, args.input.lvec);
            __syncwarp();
            if(laneid() == 0) arrive(args.inputs_finished); // done!
            max(max_lvec, args.state.lvec, lvec);
            sub(args.state.lvec, args.state.lvec, max_lvec);
            sub(lvec, lvec, max_lvec);
            exp2(args.state.lvec, args.state.lvec);
            exp2(lvec, lvec);
            add(sum_lvec, args.state.lvec, lvec);
            div(args.state.lvec, args.state.lvec, sum_lvec);
            div(lvec, lvec, sum_lvec);
            mul_row(args.state.o, args.state.o, args.state.lvec);
            mul_row(o, o, lvec);
            add(args.state.o, args.state.o, o);
            log2(sum_lvec, sum_lvec);
            add(args.state.lvec, sum_lvec, max_lvec);
        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
            if(args.common.dst.batch_idx >= 0) {
                auto &o_smem = reinterpret_cast<st_bf<16, QVO_D/8>&>(args.scratch.o[group<8>::warpid()]);
                store(o_smem, args.state.o);
                __syncwarp();
                tma::store_async(args.globals.O, o_smem, {args.common.dst.batch_idx, args.common.dst.seq_idx, 0, group<8>::warpid()});
            }
            else {
                store(args.scratch.o[group<8>::warpid()], args.state.o);
                if(group<8>::warpid() == 0) store(args.scratch.lvec, args.state.lvec);
                __syncwarp();
                tma::store_async(args.globals.O_scratch, args.scratch.o[group<8>::warpid()], {-args.common.dst.batch_idx-1, args.common.dst.seq_idx, 0, group<8>::warpid()});
                if(group<8>::warpid() == 0) tma::store_async(args.globals.Lvec_scratch, args.scratch.lvec, {-args.common.dst.batch_idx-1, args.common.dst.seq_idx, 0});
            }
            tma::store_async_wait();
            group<8>::sync(11);
            // Increment the semaphore for the next stage, if this is not the last one.
            if(args.common.dst.batch_idx < 0) {
                if(group<8>::laneid() == 0) {
                    args.globals.semaphore[{-args.common.dst.batch_idx-1, args.common.dst.seq_idx}] = args.globals.tic;
                }
            }
            if(warpgroup::laneid() == 0) arrive(args.finish_finished, WARPGROUP_WARPS); // done!
            args.timing.end = clock64();
        }
    };
};
// union task {
//     struct {
//         int type; // 1 for partial, 2 for reduction
//         int batch_id;
//         int token_ids[4];
//         int start_pos, end_pos, seq_length; // only used for partial tasks.
//         std::string name; // for debugging
//         float start, finish;
//         int processor; // -1 if not assigned
//         bool write_scratch; // is output direct or to scratch?
//     } partial;
//     struct {
//         int type; // 1 for partial, 2 for reduction
//         int batch_id;
//         int token_id;
//         std::string name; // for debugging
//         float start, finish;
//         std::vector<int> dependencies; // for reductions.
//         int processor; // -1 if not assigned
//         bool write_scratch; // is output direct or to scratch?
//     } reduction;
// };
// void make_schedule(int num_processors, std::vector<int> sequences) {
//     std::sort(sequences.begin(), sequences.end(), [](int a, int b) { return a > b; }); // sort in descending order

//     int total_partial_work = std::accumulate(sequences.begin(), sequences.end(), 0, [](int a, int b) { return a + ((b+NUM_ROWS-1) / NUM_ROWS); }); // in units of cache_tile's
//     // We'd like to have partials scheduled so that they:
//     // - Minimize the number of reductions we need to do.
//     // - Balance the load across processors.

//     // First we want to figure out how many partial's we want to actually schedule.
//     int num_

//     std::vector< std::vector<int> > schedule(num_processors);
//     for(auto seq_len : sequences) {

//     }
// }

PYBIND11_MODULE(mla_decode, m) {
    m.doc() = "mla_decode python module";
    py::bind_kernel<interpreter::kernel<config, partial_template, reduction_template>>(m, "mla_decode",
        &config::globals::instructions,
        &config::globals::Q,
        &config::globals::QV,
        &config::globals::K_cache,
        &config::globals::V_cache,
        &config::globals::Table,
        &config::globals::O,
        &config::globals::O_scratch,
        &config::globals::Lvec_scratch,
        &config::globals::semaphore,
        &config::globals::Softmax_scale,
        &config::globals::tic,
        &config::globals::timings
    );
}