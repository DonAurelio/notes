#!/usr/bin/env python3
"""
Minimal reproducer: Kineto loses ZE events when profile() opens before
DDP __init__ in a 2-rank mpiexec setup.

Usage:
    mpiexec -n 2 -ppn 1 -hosts H0,H1 ccl_local_wrap.sh python3 repro_kineto_ze.py --scope wide
    mpiexec -n 2 -ppn 1 -hosts H0,H1 ccl_local_wrap.sh python3 repro_kineto_ze.py --scope narrow
"""
import sys
import argparse
import torch
import torch.nn as nn
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


def run_step(model, device):
    x = torch.randn(64, 64, device=device)
    out = model(x)
    synchronize(device)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scope", choices=["wide", "narrow"], required=True)
    p.add_argument("--steps", type=int, default=3)
    args = p.parse_args()

    rank, world_size, local_size = get_mpi_env()
    local_rank = get_local_rank(rank, local_size)
    device = select_device("xpu", local_rank)

    activities = [ProfilerActivity.CPU, ProfilerActivity.XPU]

    # wide: profiler opens before DDP init
    if args.scope == "wide":
        prof_ctx = profile(activities=activities)
        prof_ctx.__enter__()

    initialize_process_group(rank, world_size, "xccl", device)
    model = nn.Linear(64, 64).to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    synchronize(device)

    # narrow: profiler opens after DDP init
    if args.scope == "narrow":
        prof_ctx = profile(activities=activities)
        prof_ctx.__enter__()

    for _ in range(args.steps):
        run_step(model, device)

    synchronize(device)
    prof_ctx.__exit__(None, None, None)

    import torch.distributed as dist
    dist.destroy_process_group()

    if rank == 0:
        avgs = prof_ctx.key_averages()
        n_events  = sum(e.count for e in avgs)
        ze_events = sum(e.count for e in avgs if e.key.startswith("ze"))
        print(f"scope={args.scope}  n_events={n_events}  ze_events={ze_events}")


if __name__ == "__main__":
    main()
