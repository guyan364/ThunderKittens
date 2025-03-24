#include <cuda.h>
#include "kittens.cuh"

// 128x128x128 tile size
constexpr int TileM = 128;
constexpr int TileN = 128;
constexpr int TileK = 128;

using namespace kittens;

// 8 workers, 2x4
constexpr int NumWorkers = 8;
constexpr int WorkerM = 2;
constexpr int WorkerN = 4;

constexpr int Stages = 3; // 3 stage  

// global layout
using global_ab = gl<fp8e4m3, -1, -1, -1, -1>;
using global_s = gl<float, -1, -1, -1, -1>;
using global_c = gl<bf16, -1, -1, -1, -1>;

// shared tile
using shared_a = st<fp8e4m3, TileM/WorkerM, TileK>;
using shared_b = st<fp8e4m3, TileN/WorkerN, TileK>;
using shared_as = sv<float, TileM/WorkerM>;
using shared_bs = sv<float, 16>;
using shared_c = st<bf16, TileM/WorkerM, TileN/WorkerN>;

// reg tile per worker
using reg_a = rt<fp8e4m3, TileM/WorkerM, TileK>;
using reg_b = rt<fp8e4m3, TileN/WorkerN, TileK>;
using reg_c = rt<float, TileM / WorkerM, TileN / WorkerN>;
using reg_out_c = rt<bf16, TileM / WorkerM, TileN / WorkerN>;
using reg_as = reg_c::col_vec; // use reg c tile col vec

struct Globals {
  global_ab A, B;
  global_s AS, BS;
  global_c C;
};

__device__ inline int worker_idx_m(int warp_id) {
  return warp_id / WorkerN;
}

__device__ inline int worker_idx_n(int warp_id) {
  return warp_id % WorkerN;
}

__device__ inline int cdiv(int a, int b) {
  return (a + b - 1) / b;
}


__device__ inline void load_async_float(void *dst, void *src) {
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
      :: "r"(smem_int_ptr),
          "l"(src),
          "n"(4));
}

__device__ inline void commit_cp_async() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__launch_bounds__(NumWorkers*WARP_THREADS, 1)
__global__ void deepgemm_kernel(const __grid_constant__ Globals g) {
  using load_a_group = group<NumWorkers / WorkerM>;
  using load_b_group = group<NumWorkers / WorkerN>;

  extern __shared__ alignment_dummy __shm[];
  shared_allocator al((int*)&__shm[0]);

  const int bidx = blockIdx.y;
  const int bidy = blockIdx.x;
  const int w_idx = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;

  // worker idx
  const int w_idx_m = worker_idx_m(w_idx);
  const int w_idx_n = worker_idx_n(w_idx);

  const int load_a_id = load_a_group::groupid();
  const int load_b_id = load_b_group::groupid();

  shared_a (&a_smem)[WorkerM][Stages] = al.allocate<shared_a, WorkerM, Stages>();
  shared_b (&b_smem)[WorkerN][Stages] = al.allocate<shared_b, WorkerN, Stages>();
  shared_as (&as_smem)[WorkerM][Stages] = al.allocate<shared_as, WorkerM, Stages>();
  shared_bs (&bs_smem)[Stages] = al.allocate<shared_bs, Stages>();

  reg_a a_reg;
  reg_b b_reg;
  reg_c c_reg, tmp_accum;
  reg_as as_reg;
  float bs_reg;
  c_reg = 0.f;
  tmp_accum = 0.f;

  const int k_blocks = cdiv(g.A.cols(), TileK);

  int cur_load_stage = 0;
  int cur_load_ktile = 0;
  int cur_mma_stage = 0;

  auto load_scale = [&] (int load_stage, int load_tile) {
    if (w_idx == 0) { // only warp 0 load scales
      for (int j = lane_id; j < TileM; j += 32) {
        int global_m = bidx * TileM + j;
        int work_id = j / (TileM / WorkerM);
        int global_n = load_tile;
        if (global_m < g.A.rows()) {
          load_async_float(
            &as_smem[work_id][load_stage][j % (TileM / WorkerM)],
            &g.AS[{global_m, global_n}]
          );
        }
      }
      if (lane_id == 0) {
        load_async_float(
          &bs_smem[cur_load_stage][0],
          &g.BS[{bidy, load_tile}]
        );
      }
      __syncwarp();
    }
  };

  for(int i = 0; i < Stages - 1; ++i) {
    if (i < k_blocks) {
      load_a_group::load_async<2, false, false>(
        a_smem[load_a_id][cur_load_stage],
        g.A,
        {0, 0, bidx * WorkerM + load_a_id, i}
      );
      load_b_group::load_async<2, false, false>(
        b_smem[load_b_id][cur_load_stage],
        g.B,
        {0, 0, bidy * WorkerN + load_b_id, i}
      );
      load_scale(cur_load_stage, i);
    }
    ++cur_load_stage;
    ++cur_load_ktile;
    commit_cp_async();
  }


  for (int k_block = 0; k_block < k_blocks; ++k_block) {
    load_async_wait<Stages - 2>();
    __syncthreads();
    load(a_reg, a_smem[w_idx_m][cur_mma_stage]);
    load(b_reg, b_smem[w_idx_n][cur_mma_stage]);
    bs_reg = bs_smem[cur_mma_stage][0];
    load(as_reg, as_smem[w_idx_m][cur_mma_stage]);
    as_reg *= bs_reg;
    mma<transpose::N, transpose::T>(tmp_accum, a_reg, b_reg, tmp_accum); 
    mul_row(tmp_accum, tmp_accum, as_reg);
    c_reg += tmp_accum;
    tmp_accum = 0.f;

    if (cur_load_ktile < k_blocks) {
      load_a_group::load_async<2, false, false>(
        a_smem[load_a_id][cur_load_stage],
        g.A,
        {0, 0, bidx * WorkerM + load_a_id, cur_load_ktile}
      );
      load_b_group::load_async<2, false, false>(
        b_smem[load_b_id][cur_load_stage],
        g.B,
        {0, 0, bidy * WorkerN + load_b_id, cur_load_ktile}
      );
      load_scale(cur_load_stage, cur_load_ktile);
      ++cur_load_ktile;
      cur_load_stage = (cur_load_stage + 1) % Stages;
    }
    commit_cp_async();
    cur_mma_stage = (cur_mma_stage + 1) % Stages;
  }


  // reuse shared memory
  __syncthreads();
  shared_allocator bl((int*)&__shm[0]);
  shared_c (&c_smem)[WorkerM][WorkerN] = bl.allocate<shared_c, WorkerM, WorkerN>();
  store(c_smem[w_idx_m][w_idx_n], c_reg);
  store<2, false>(g.C, c_smem[w_idx_m][w_idx_n], {0, 0, bidx * WorkerM + w_idx_m, bidy * WorkerN + w_idx_n});
}

extern "C" {
  void deepgemm(
    int m, int n, int k, 
    void *A, void *AS, void *B, void *BS, void *C
  ) {
    assert(m % 128 == 0);
    assert(n % 128 == 0);
    assert(k % 128 == 0);
    Globals g = {
      global_ab((fp8e4m3*)A, 1, 1, m, k),
      global_ab((fp8e4m3*)B, 1, 1, n, k),
      global_s((float*)AS, 1, 1, m, k/128),
      global_s((float*)BS, 1, 1, n/128, k/128),
      global_c((bf16*)C, 1, 1, m, n)
    };
    cudaFuncSetAttribute(deepgemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 100000);
    dim3 grid_shape(n / 128, m / 128);
    dim3 cta_shape(NumWorkers * WARP_THREADS);
    deepgemm_kernel<<<grid_shape, cta_shape, 100000>>>(g);
  }
}