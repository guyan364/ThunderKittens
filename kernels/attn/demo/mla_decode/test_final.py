import mla_decode
import torch
import numpy as np
import math
import heapq
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

m1 = 0.0454
b1 = 10
m2 = 0.366
b2 = 8

# ----------------------------------------------------------------------
# Scheduling code (task generation, scheduling, and argument creation)
# ----------------------------------------------------------------------
@dataclass
class Task:
    uid: int
    batch_id: int              # Which sequence this task belongs to.
    tok_ids: List[int]         # Query indices
    name: str
    task_type: str             # "partial" or "reduction"
    duration: float
    dependencies: List[int] = field(default_factory=list)
    start: float = None
    finish: float = None
    processor: int = None
    args: dict = field(default_factory=dict)

def generate_sequence_tasks(batch_id: int, length: int, chunk_pages: int,
                            m1: float, b1: float, b2: float,
                            starting_id: int = 0, new_tokens: int = 1,
                            page_size: int = 256) -> Tuple[List[Task], int]:
    """
    Generates tasks for one sequence.
      - Creates PARTIAL tasks that process contiguous chunks.
      - Then builds MERGE tasks (reductions) in a tree-like reduction until one result remains.
    
    PARTIAL cost = m1 * (chunk length) + b1.
    MERGE cost = b2.
    
    Each task is annotated with batch_id.
    Returns a list of tasks for this sequence and the next available task uid.
    """
    chunk_size = chunk_pages * page_size
    tasks = []
    partial_tasks = [[] for _ in range((new_tokens+3)//4)]
    # Create PARTIAL tasks (with no dependencies)
    for i in range(0, length, chunk_size):
        chunk_length = min(chunk_size, length - i)
        duration = m1 * (((chunk_length+31)//32)*32) + b1
        for n in range((new_tokens+3)//4):
            tok_ids = list(range(4*n, min(4*n+4, new_tokens)))
            task = Task(
                uid=starting_id,
                batch_id=batch_id,
                name=f"Seq{batch_id}_Partial_{i}-{i+chunk_length}_tok{tok_ids[0]}-{tok_ids[-1]}",
                task_type="partial",
                duration=duration,
                dependencies=[],
                tok_ids=tok_ids,
                args={"start": i,
                      "end": i+chunk_length,
                      "write_scratch": (chunk_length != length),
                      "length": length}
            )
            tasks.append(task)
            partial_tasks[n].append(task)
            starting_id += 1

    for new_token in range(new_tokens):
        TREE_BASE = 12
        n_idx = new_token // 4
        merge_tasks = []
        for i in range(0, len(partial_tasks[n_idx])-1, TREE_BASE):
            dep = [partial_tasks[n_idx][j].uid for j in range(i, min(i+TREE_BASE, len(partial_tasks[n_idx])))]
            if i+TREE_BASE+1 == len(partial_tasks[n_idx]):
                dep.append(partial_tasks[n_idx][i+TREE_BASE].uid)
            merge_tasks.append(Task(
                uid=starting_id,
                batch_id=batch_id,
                name=f"Seq{batch_id}_Merge_{i//2}_tok{new_token}",
                task_type="reduction",
                duration=b2 + m2*(len(dep)-1),
                dependencies=dep,
                tok_ids=[new_token],
                args={"write_scratch": (i != 0)}
            ))
            starting_id += 1
        tasks.extend(merge_tasks)
        while len(merge_tasks) > 0:
            next_merge_tasks = []
            for i in range(0, len(merge_tasks)-1, TREE_BASE):
                new_dep = [merge_tasks[j].uid for j in range(i+1, min(i+TREE_BASE, len(merge_tasks)))]
                if i+TREE_BASE+1 == len(merge_tasks):
                    new_dep.append(merge_tasks[i+TREE_BASE].uid)
                merge_tasks[i].dependencies.extend(new_dep)
                merge_tasks[i].duration = b2 + m2*(len(merge_tasks[i].dependencies)-1)
                next_merge_tasks.append(merge_tasks[i])
            merge_tasks = next_merge_tasks
    return tasks, starting_id

def visualize_schedule(tasks: List[Task], num_processors: int):
    """
    Plots a Gantt chart of the scheduled tasks.
    
    Tasks are grouped by processor; each task is shown as a horizontal bar.
    Colors distinguish between 'partial' (blue) and 'reduction' (red) instructions.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {
        'partial': 'skyblue',
        'reduction': 'salmon'
    }
    proc_tasks: Dict[int, List[Task]] = {i: [] for i in range(num_processors)}
    for task in tasks:
        proc_tasks[task.processor].append(task)
    height = 9
    for proc_id, tasks_on_proc in proc_tasks.items():
        y_coord = proc_id * 10
        for task in tasks_on_proc:
            ax.broken_barh(
                [(task.start, task.finish-task.start)],
                (y_coord, height),
                facecolors=colors.get(task.task_type, 'gray')
            )
            ax.text((task.start + task.finish)/2, y_coord + height/2,
                    task.name, ha='center', va='center', fontsize=8, color='black')
    ax.set_xlabel("Time")
    ax.set_ylabel("Processors")
    ax.set_yticks([i * 10 + height/2 for i in range(num_processors)])
    ax.set_yticklabels([f"Processor {i}" for i in range(num_processors)])
    ax.set_title("Gantt Chart of Scheduled Tasks (Priority Scheduler)")
    plt.tight_layout()
    plt.savefig(f"gantt_{int(time.time())}.png")
    
def priority_schedule_tasks(tasks: List[Task], num_processors: int) -> List[Task]:
    """
    Schedules tasks on available processors using a custom heuristic.
    Updates each task's start, finish, and processor.
    """
    tasks_by_id: Dict[int, Task] = {t.uid: t for t in tasks}
    in_degree: Dict[int, int] = {t.uid: len(t.dependencies) for t in tasks}
    dependents: Dict[int, List[int]] = {t.uid: [] for t in tasks}
    for t in tasks:
        for dep in t.dependencies:
            dependents[dep].append(t.uid)

    # Compute total work per sequence.
    seq_total_work: Dict[int, float] = {}
    for t in tasks:
        seq_total_work.setdefault(t.batch_id, 0)
        seq_total_work[t.batch_id] += t.duration

    ready = [t for t in tasks if in_degree[t.uid] == 0]
    processors: List[Tuple[float, int]] = [(0, pid) for pid in range(num_processors)]
    heapq.heapify(processors)

    while ready:
        current_time, proc_id = heapq.heappop(processors)
        ready_partial = [t for t in ready if t.task_type == "partial"]
        ready_merge   = [t for t in ready if t.task_type == "reduction"]

        filtered_merge = [t for t in ready_merge if current_time+1 >= tasks_by_id[t.dependencies[0]].finish]
        if filtered_merge:
            filtered_merge.sort(key=lambda t: (-t.duration, t.uid))
            chosen_task = filtered_merge[0]
        elif ready_partial:
            ready_partial.sort(key=lambda t: (-seq_total_work[t.batch_id], t.uid))
            chosen_task = ready_partial[0]
        elif ready_merge:
            ready_merge.sort(key=lambda t: (-t.duration, t.uid))
            chosen_task = ready_merge[0]
        else:
            chosen_task = ready[0]

        ready.remove(chosen_task)
        if chosen_task.task_type == "partial":
            earliest_start = max([0] + [tasks_by_id[dep].finish for dep in chosen_task.dependencies])
            start_time = max(current_time, earliest_start)
            finish_time = start_time + chosen_task.duration
        else:
            start_time = max(current_time, tasks_by_id[chosen_task.dependencies[0]].finish)
            finish_time = max([max(start_time + b2, tasks_by_id[chosen_task.dependencies[i]].finish) + (len(chosen_task.dependencies)-i)*m2
                               for i in range(1, len(chosen_task.dependencies))])
        chosen_task.start = start_time
        chosen_task.finish = finish_time
        chosen_task.processor = proc_id

        heapq.heappush(processors, (finish_time, proc_id))
        for dep_id in dependents[chosen_task.uid]:
            in_degree[dep_id] -= 1
            if in_degree[dep_id] == 0:
                ready.append(tasks_by_id[dep_id])
    if any(in_degree[t.uid] != 0 for t in tasks):
        raise ValueError("Cycle detected in task dependencies!")
    return tasks

def create_arguments_from_task_schedule(tasks: List[Task], new_tokens: int, num_processors: int = 1):
    OP_PARTIAL, OP_REDUCTION = 1, 2

    def make_partial_arg(task: Task) -> List[int]:
        return [OP_PARTIAL,
                task.uid,  # uid
                -task.uid-1 if task.args["write_scratch"] else task.batch_id,  # destination (negative means write scratch)
                min(task.tok_ids),  # start token
                task.batch_id,
                min(task.tok_ids),  # duplicate start token
                task.args["start"], task.args["end"], task.args["length"]] + [0]*23

    def make_merge_arg(task: Task) -> List[int]:
        assert(len(task.dependencies) <= 11+16)
        args_list = [OP_REDUCTION,
                     task.uid,  # uid
                     len(task.dependencies)-1,  # number of dependencies minus one
                     -task.uid-1 if task.args["write_scratch"] else task.batch_id,
                     task.tok_ids[0]] + task.dependencies
        return args_list + [0]*(32 - len(args_list))

    num_instructions = max(t.uid for t in tasks) + 1
    processor_tasks = [[] for _ in range(num_processors)]
    for task in tasks:
        processor_tasks[task.processor].append(task)
    for pid in range(num_processors):
        processor_tasks[pid].sort(key=lambda t: t.start)
    print('Final finish time:', max(t.finish for t in tasks))
    print('Max number of dependencies:', max(len(t.dependencies) for t in tasks))
    max_num_processor_instructions = max(len(ptasks) for ptasks in processor_tasks)
    Instructions = torch.zeros((num_processors, max_num_processor_instructions, 32), dtype=torch.int32, device='cpu')
    O_scratch = torch.zeros((num_instructions, new_tokens, 16, 512), dtype=torch.float32, device='cpu')
    L_scratch = torch.zeros((num_instructions, new_tokens, 16), dtype=torch.float32, device='cpu')
    Semaphore = torch.zeros((num_instructions, new_tokens), dtype=torch.int32, device='cpu')
    Timings = torch.zeros((num_processors, max_num_processor_instructions, 2), dtype=torch.int32, device='cpu')
    for pid in range(num_processors):
        for tid, task in enumerate(processor_tasks[pid]):
            if task.task_type == "partial":
                Instructions[pid, tid, :] = torch.tensor(make_partial_arg(task), dtype=torch.int32, device='cpu')
            elif task.task_type == "reduction":
                Instructions[pid, tid, :] = torch.tensor(make_merge_arg(task), dtype=torch.int32, device='cpu')
    if torch.cuda.is_available():
        return (Instructions.cuda(), O_scratch.cuda(),
                L_scratch.cuda(), Semaphore.cuda(),
                Timings.cuda())
    return Instructions, O_scratch, L_scratch, Semaphore, Timings

def sample_schedule_generator(page_size: int = 256, new_tokens: int = 1, lengths: List[int] = None, partial_pages: List[int] = None, table: list = None) -> List[Task]:
    """
    For demonstration, we schedule one sequence (batch 0) with a specified length.
    Using new_tokens=1 yields one partial task (and no merge tasks).
    The page table is passed in dynamically (if not provided, a default is used).
    """
    if partial_pages is None:
        partial_pages = [1 if length < 8192 else 2 for length in lengths]
    tasks = []
    next_task_id = 0
    # One sequence: (batch_id, length, chunk_pages)
    sequences = [
        (i, length, pp) for i, (length, pp) in enumerate(zip(lengths, partial_pages))
    ]
    for batch_id, seq_length, chunk_pages in sequences:
        seq_tasks, next_task_id = generate_sequence_tasks(
            new_tokens=new_tokens,
            page_size=page_size,
            batch_id=batch_id,
            length=seq_length,
            chunk_pages=chunk_pages,
            m1=m1, b1=b1, b2=b2,
            starting_id=next_task_id
        )
        tasks.extend(seq_tasks)
    return tasks

# ----------------------------------------------------------------------
# Main test script (integrating the reference page table creation)
# ----------------------------------------------------------------------
def main(length: int = 65536):
    D_QK, D_VO, D_QRot = 576, 512, 64
    PAGE_SIZE = 256
    B, NEW_TOKENS, H = 1, 4, 16  # single token (naive single partial op)
    MAX_LENGTH = 65536
    LENGTH = length               # sequence length
    NUM_PAGES = 1000             # number of pages in cache
    NUM_PROCESSORS = 132         # number of processors

    torch.manual_seed(0)

    T = MAX_LENGTH // PAGE_SIZE
    table_tensor = torch.zeros(B, T, dtype=torch.int32).cuda()
    lengths = torch.full((B,), LENGTH, dtype=torch.int32, device='cuda')
    sizes = (lengths + (PAGE_SIZE - 1)) // PAGE_SIZE
    sequence_ids, pos_ids = (
        torch.arange(T, dtype=torch.int32, device='cuda')[None, :].expand(B, -1)
        .lt(sizes.view(-1, 1))
        .nonzero(as_tuple=True)
    )
    randperm = torch.randperm(NUM_PAGES, device='cuda')[:(LENGTH+PAGE_SIZE-1)//PAGE_SIZE].sort().values.int()
    table_tensor[sequence_ids, pos_ids] = randperm

    # tasks = sample_schedule_generator(page_size=PAGE_SIZE, new_tokens=NEW_TOKENS, lengths=[4671, 45096, 1750, 1701], partial_pages=[1, 1, 1, 1])
    # tasks = sample_schedule_generator(page_size=PAGE_SIZE, new_tokens=NEW_TOKENS, lengths=[LENGTH*1//2], partial_pages=[1])
    tasks = sample_schedule_generator(page_size=PAGE_SIZE, new_tokens=NEW_TOKENS, lengths=[LENGTH], partial_pages=[1])
    
    scheduled_tasks = priority_schedule_tasks(tasks, num_processors=NUM_PROCESSORS)
    visualize_schedule(scheduled_tasks, num_processors=NUM_PROCESSORS)
    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_arguments_from_task_schedule(
        scheduled_tasks, NEW_TOKENS, num_processors=NUM_PROCESSORS
    )
    Table = table_tensor

    cache = torch.zeros((NUM_PAGES, PAGE_SIZE, 1, D_QK), dtype=torch.bfloat16).cuda()
    total = LENGTH  # for one sequence
    latent = (torch.randn((total, 1, D_QK), dtype=torch.bfloat16).cuda() * 10)

    expanded = latent.expand(total, H, D_QK)
    maximum = LENGTH
    padded_key = torch.zeros((B, maximum, H, D_QK), dtype=torch.bfloat16).cuda()
    padded_value = torch.zeros((B, maximum, H, D_VO), dtype=torch.bfloat16).cuda()
    seq_ids, pos_ids = (
        torch.arange(maximum, device='cuda')[None, :].expand(B, -1)
        .lt(torch.tensor([LENGTH], device='cuda').view(-1, 1))
        .nonzero(as_tuple=True)
    )

    # cache from latent using the page table.
    entry_ids  = table_tensor[seq_ids, pos_ids.floor_divide(PAGE_SIZE)]
    column_ids = pos_ids.fmod(PAGE_SIZE)
    cache[entry_ids, column_ids] = latent / math.sqrt(D_QK)

    query = torch.randn((B, NEW_TOKENS, H, D_QK), dtype=torch.bfloat16).cuda() / math.sqrt(D_QK)

    O = torch.zeros((B, NEW_TOKENS, H, D_VO), dtype=torch.bfloat16).cuda().contiguous()

    softmax_scale = 1.0 / math.sqrt(D_QK)
    
    cache_view = cache.view(B, NUM_PAGES, PAGE_SIZE, D_QK)

    '''
    Changes to the interface:
    - QRot is now (B, NEW_TOKENS, H, D_QRot)
    - QV is now (B, NEW_TOKENS, H, D_VO)
    - K_cache is now (B, NUM_PAGES, PAGE_SIZE, D_QRot)
    - V_cache is now (B, NUM_PAGES, PAGE_SIZE, D_VO)
    '''
    query_rot = query[..., -D_QRot:].contiguous()
    query_v = query[..., :-D_QRot].contiguous()
    K_cache = cache_view[..., -D_QRot:].contiguous()
    V_cache = cache_view[..., :-D_QRot].contiguous()

    print("Launching MLA decode kernel...")
    mla_decode.mla_decode(Instructions, query_rot, query_v,
                          K_cache, V_cache,
                          Table, O, O_scratch, Lvec_scratch, Semaphore,
                          softmax_scale, 1, Timings)
    torch.cuda.synchronize()
    print("Kernel execution finished.")
    # print("Kernel output O:\n", O)
    O1 = O.clone()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    num_iters = 1000
    print(f"\nTiming kernel over {num_iters} iterations...")
    
    # Warmup
    for _ in range(4):
        mla_decode.mla_decode(Instructions, query_rot, query_v,
                            K_cache, V_cache,
                            Table, O, O_scratch, Lvec_scratch, Semaphore,
                            softmax_scale, _%2, Timings)
        print(f'ran warmup kernel #{_}')

        torch.cuda.synchronize()
    start_event.record()
    
    for _ in range(num_iters):
        mla_decode.mla_decode(Instructions, query_rot, query_v,
                            K_cache, V_cache,
                            Table, O, O_scratch, Lvec_scratch, Semaphore,
                            softmax_scale, _%2, Timings)
    
    torch.cuda.synchronize()
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time = start_event.elapsed_time(end_event)
    avg_time = elapsed_time / num_iters
    print(f"Average kernel execution time: {avg_time*1000:.3f} us")
    print(f"Total time for {num_iters} iterations: {elapsed_time*1000:.3f} us\n")

    bounds = (torch.arange(NEW_TOKENS, dtype=torch.int32, device='cuda')[None, :] +
              lengths[:, None] - NEW_TOKENS)
    mask = (torch.arange(maximum, dtype=torch.int32, device='cuda')[None, None, None, :]
            .expand(B, H, NEW_TOKENS, -1)
            .le(bounds[:, None, :, None].expand(B, H, NEW_TOKENS, 1)))

    from torch.nn.functional import scaled_dot_product_attention as sdpa
    
    padded_key[seq_ids, pos_ids] = expanded / math.sqrt(D_QK)
    padded_value[seq_ids, pos_ids] = expanded[..., :D_VO] / math.sqrt(D_QK)
    
    ref = sdpa(
        query=query.transpose(1, 2).float(),
        key=padded_key.transpose(1, 2).float(),
        value=padded_value.transpose(1, 2).float(),
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        scale=softmax_scale,
        enable_gqa=False,
    ).transpose(1, 2)

    print("Testing correctness...")
    diffs = []
    for _ in range(num_iters * 10):
        mla_decode.mla_decode(Instructions, query_rot, query_v,
                            K_cache, V_cache,
                            Table, O, O_scratch, Lvec_scratch, Semaphore,
                            softmax_scale, _%2, Timings)
        max_diff = torch.abs(O - ref).max()
        mean_diff = torch.abs(O - ref).mean()
        diffs.append((max_diff, mean_diff))

    # print("Reference output:\n", ref)
    print("ref mean:", torch.mean(ref.abs()))
    print("Kernel output mean:", torch.mean(O.abs()))
    print("Max absolute diff:", torch.max(torch.abs(O - ref)))
    print("Avg absolute diff:", torch.mean(torch.abs(O - ref)))
    print("Initial kernel output mean:", torch.mean(O1.abs()))
    print("Initial Max absolute diff:", torch.max(torch.abs(O1 - ref)))
    print("Initial Avg absolute diff:", torch.mean(torch.abs(O1 - ref)))

    diffs_tensor = torch.tensor(diffs)
    print(f"Max of max diffs over {num_iters*10} iterations:", diffs_tensor[:, 0].max())
    print(f"Mean of mean diffs over {num_iters*10} iterations:", diffs_tensor[:, 1].mean())
    print(f"Num iterations with nans: {((diffs_tensor[:, 0].isnan() | diffs_tensor[:, 1].isnan()).sum())}")

    # Convert cycles to microseconds (1.8 GHz = 1800 MHz = 1.8 cycles/ns = 0.0018 cycles/us)
    timings_us = Timings.float() / 1800

    # Get unique instruction types for coloring
    instruction_types = Instructions[:,:,0].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(instruction_types)))
    color_map = dict(zip(instruction_types.cpu().numpy(), colors))

    # Create Gantt chart
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # For each processor
    for proc in range(Timings.shape[0]):
        for instr in range(Timings.shape[1]):
            start = timings_us[proc, instr, 0].item()
            duration = timings_us[proc, instr, 1].item() - start
            if duration > 0:  # Only plot if there was actual work
                instr_type = Instructions[proc,instr,0].item()
                ax.barh(proc, duration, left=start, 
                       color=color_map[instr_type],
                       alpha=0.7)

    # Customize the chart
    ax.set_xlabel('Time (microseconds)')
    ax.set_ylabel('Processor ID')
    ax.set_title('Instruction Execution Timeline')

    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_map[t.item()], 
                      label=f'Instruction {t.item()}')
                      for t in instruction_types]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(f'instruction_timeline_{int(time.time())}.png')

    # Print timing statistics
    print("\nTiming Statistics (microseconds):")
    print(f"Average instruction time: {timings_us.mean():.3f}")
    print(f"Max instruction time: {timings_us.max():.3f}")
    print(f"Total execution time: {timings_us.sum():.3f}")
    print("\nBy instruction type:")
    for instr_type in instruction_types:
        mask = Instructions[:,:,0] == instr_type
        mean_time = timings_us[mask].mean()
        print(f"Instruction {instr_type.item()}: {mean_time:.3f}")

    breakpoint()

if __name__ == '__main__':
    import sys
    length = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
    main(length)
