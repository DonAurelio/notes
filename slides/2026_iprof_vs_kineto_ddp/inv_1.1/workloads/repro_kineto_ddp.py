#!/usr/bin/env python3
"""
Minimal reproducer: Kineto XPU listener fails to attach when profile()
opens before DistributedDataParallel.__init__() under mpiexec + xccl.

Self-contained — no mpi4py, no project dependencies.
Works on a single node: mpiexec -n 2 -ppn 2 -hosts $HOST

Usage:
    # broken — ze_events ~ 0
    MASTER_ADDR=$HOST mpiexec -n 2 -ppn 2 python3 repro_kineto_ddp.py --scope wide

    # correct — ze_events > 0
    MASTER_ADDR=$HOST mpiexec -n 2 -ppn 2 python3 repro_kineto_ddp.py --scope narrow
"""
import os
import datetime
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.profiler import profile, ProfilerActivity

# ── rank info from mpiexec PALS env vars ─────────────────────────────────────

rank = int(
    os.environ.get("PALS_RANKID") or
    os.environ.get("PMI_RANK") or
    os.environ.get("OMPI_COMM_WORLD_RANK") or "0"
)
world_size = int(
    os.environ.get("PALS_WORLD_SIZE") or
    os.environ.get("PMI_SIZE") or
    os.environ.get("OMPI_COMM_WORLD_SIZE") or "1"
)
local_rank = int(
    os.environ.get("PALS_LOCAL_RANKID") or
    os.environ.get("PMI_LOCAL_RANK") or
    os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK") or str(rank)
)

os.environ.setdefault("MASTER_PORT", "2345")

torch.xpu.set_device(local_rank)
device = torch.device("xpu", local_rank)

# ── args ─────────────────────────────────────────────────────────────────────

p = argparse.ArgumentParser()
p.add_argument("--scope", choices=["wide", "narrow"], required=True,
               help="wide: profile before DDP init (broken); narrow: after (correct)")
p.add_argument("--steps", type=int, default=3)
args = p.parse_args()

activities = [ProfilerActivity.CPU, ProfilerActivity.XPU]

# ── wide: profile opens before DDP init ──────────────────────────────────────
if args.scope == "wide":
    prof = profile(activities=activities)
    prof.__enter__()

dist.init_process_group(
    backend="xccl",
    init_method="env://",
    world_size=world_size,
    rank=rank,
    timeout=datetime.timedelta(seconds=120),
)
model = nn.Linear(64, 64).to(device)
model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
torch.xpu.synchronize()

# ── narrow: profile opens after DDP init ─────────────────────────────────────
if args.scope == "narrow":
    prof = profile(activities=activities)
    prof.__enter__()

for _ in range(args.steps):
    x = torch.randn(64, 64, device=device)
    model(x)
    dist.all_reduce(torch.ones(1, device=device))
    torch.xpu.synchronize()

torch.xpu.synchronize()
prof.__exit__(None, None, None)
dist.destroy_process_group()

if rank == 0:
    avgs      = prof.key_averages()
    n_events  = sum(e.count for e in avgs)
    ze_events = sum(e.count for e in avgs if e.key.startswith("ze"))
    print(f"scope={args.scope}  n_events={n_events}  ze_events={ze_events}")
