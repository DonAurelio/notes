#!/usr/bin/env python3
"""
Minimal bisect reproducer: which call unblocks the Kineto XPU listener?

Self-contained — no mpi4py, no llama3_demo_common.
Reads rank/world_size from PALS env vars set by mpiexec.
MASTER_ADDR must be set to rank-0 hostname before launching (see run script).

Checkpoints (profile opens just after each):
  cp=0  before dist.init_process_group      [known broken]
  cp=1  after  dist.init_process_group
  cp=2  after  DistributedDataParallel.__init__
  cp=3  after  first all_reduce
  cp=4  after  first matmul                 [known working]

Usage:
    mpiexec -n 2 -ppn 1 -hosts H0,H1 python3 repro_bisect_mpi4py.py --checkpoint N
"""
import os
import datetime
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.profiler import profile, ProfilerActivity

# ── rank / world size from mpiexec PALS env ──────────────────────────────────

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

# MASTER_ADDR must be set in the environment before mpiexec (export MASTER_ADDR=$HOST0)
os.environ.setdefault("MASTER_PORT", "2345")

torch.xpu.set_device(local_rank)
device = torch.device("xpu", local_rank)

# ── args ─────────────────────────────────────────────────────────────────────

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=int, choices=[0, 1, 2, 3, 4], required=True)
p.add_argument("--steps", type=int, default=3)
args = p.parse_args()

cp         = args.checkpoint
activities = [ProfilerActivity.CPU, ProfilerActivity.XPU]
prof       = None

LABELS = {
    0: "before init_process_group",
    1: "after  init_process_group",
    2: "after  DDP.__init__",
    3: "after  first all_reduce",
    4: "after  first matmul",
}

def maybe_open(at):
    global prof
    if prof is None and cp == at:
        prof = profile(activities=activities)
        prof.__enter__()

# ── cp=0: before everything ──────────────────────────────────────────────────
maybe_open(0)

dist.init_process_group(
    backend="xccl",
    init_method="env://",
    world_size=world_size,
    rank=rank,
    timeout=datetime.timedelta(seconds=120),
)

# ── cp=1: after init_process_group ───────────────────────────────────────────
maybe_open(1)

model = nn.Linear(64, 64).to(device)
model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
torch.xpu.synchronize()

# ── cp=2: after DDP.__init__ ─────────────────────────────────────────────────
maybe_open(2)

dist.all_reduce(torch.ones(1, device=device))
torch.xpu.synchronize()

# ── cp=3: after first all_reduce ─────────────────────────────────────────────
maybe_open(3)

x = torch.randn(64, 64, device=device)
torch.matmul(x, x)
torch.xpu.synchronize()

# ── cp=4: after first matmul ─────────────────────────────────────────────────
maybe_open(4)

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
    print(f"cp={cp} ({LABELS[cp]})  n_events={n_events}  ze_events={ze_events}")
