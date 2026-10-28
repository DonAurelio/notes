#!/usr/bin/env python3
"""
Bisect which call initializes ZE in a way the Kineto XPU listener cannot hook into.

Five checkpoints — profile() opens at each one in turn, wrapping the rest of the
run. All checkpoints see the same step loop (matmul + all_reduce). Compare ze_events:
the first checkpoint that gives ze_events > 0 is the call that unblocks ZE.

Checkpoint 0: before dist.init_process_group      (reference — known broken)
Checkpoint 1: after  dist.init_process_group
Checkpoint 2: after  DistributedDataParallel.__init__
Checkpoint 3: after  first all_reduce
Checkpoint 4: after  first torch.matmul            (reference — known working)

Usage:
    mpiexec -n 2 -ppn 1 -hosts H0,H1 ccl_local_wrap.sh \\
        python3 repro_bisect.py --checkpoint N --steps 3
"""
import sys
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent / "workloads"))
from llama3_demo_common import (
    get_local_rank,
    get_mpi_env,
    initialize_process_group,
    select_device,
    synchronize,
)

CHECKPOINTS = {
    0: "before init_process_group",
    1: "after  init_process_group",
    2: "after  DDP.__init__",
    3: "after  first all_reduce",
    4: "after  first matmul",
}


def open_prof(cp, target, activities):
    """Return (prof_ctx or None). Opens and enters profiler at the target checkpoint."""
    if cp == target:
        ctx = profile(activities=activities)
        ctx.__enter__()
        return ctx
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=int, choices=list(CHECKPOINTS), required=True,
                   help="Open profiler at this checkpoint (0–4)")
    p.add_argument("--steps", type=int, default=3)
    args = p.parse_args()

    cp = args.checkpoint
    activities = [ProfilerActivity.CPU, ProfilerActivity.XPU]

    rank, world_size, local_size = get_mpi_env()
    local_rank = get_local_rank(rank, local_size)
    device = select_device("xpu", local_rank)

    prof = None

    # ── checkpoint 0 ────────────────────────────────────────────────────────
    prof = prof or open_prof(cp, 0, activities)

    initialize_process_group(rank, world_size, "xccl", device)

    # ── checkpoint 1 ────────────────────────────────────────────────────────
    prof = prof or open_prof(cp, 1, activities)

    model = nn.Linear(64, 64).to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    synchronize(device)

    # ── checkpoint 2 ────────────────────────────────────────────────────────
    prof = prof or open_prof(cp, 2, activities)

    # first all_reduce (outside step loop so we can open profiler after it)
    dist.all_reduce(torch.ones(1, device=device))
    synchronize(device)

    # ── checkpoint 3 ────────────────────────────────────────────────────────
    prof = prof or open_prof(cp, 3, activities)

    # first matmul (outside step loop)
    x = torch.randn(64, 64, device=device)
    torch.matmul(x, x)
    synchronize(device)

    # ── checkpoint 4 ────────────────────────────────────────────────────────
    prof = prof or open_prof(cp, 4, activities)

    # remaining steps
    for _ in range(args.steps):
        x = torch.randn(64, 64, device=device)
        out = model(x)
        dist.all_reduce(torch.ones(1, device=device))
        synchronize(device)

    synchronize(device)
    prof.__exit__(None, None, None)
    dist.destroy_process_group()

    if rank == 0:
        avgs = prof.key_averages()
        n_events  = sum(e.count for e in avgs)
        ze_events = sum(e.count for e in avgs if e.key.startswith("ze"))
        label = CHECKPOINTS[cp]
        print(f"cp={cp} ({label})  n_events={n_events}  ze_events={ze_events}")


if __name__ == "__main__":
    main()
