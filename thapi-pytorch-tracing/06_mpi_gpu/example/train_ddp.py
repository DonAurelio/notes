#!/usr/bin/env python
"""Step 5 — DistributedDataParallel: the standard multi-process training pattern.

N ranks, each a model replica. mpiexec spawns the ranks; torch.distributed wires
them into a process group (gloo backend on CPU). Every backward() triggers an
allreduce to average gradients across ranks -- this is real data-parallel SGD,
the way multi-GPU/multi-node training is actually done.

This single workload feeds the whole stage:
  * focus #1 (coherence): does each rank yield its own coherent vpid trace?
  * focus #2 (collectives): do the DDP allreduce/broadcast ops reach the hook?
  * focus #3 (launch vs comm): what does a collective's entry/exit interval mean?
  * focus #4 (thread structure): autograd engine + DDP reducer threads per rank.

Rank/world come from the launcher (PALS sets PMI_RANK / PMI_SIZE).

Env:
  DIM      layer width               (default 256)
  STEPS    training steps            (default 3)
  DEVICE   cpu (default) | xpu       (xpu needs a ccl backend + compute node)
  BACKEND  gloo (default) | ccl | mpi
"""
import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class Net(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim)

    def forward(self, x):
        return torch.relu(self.l2(torch.relu(self.l1(x))))


def env_rank():
    # Cray PALS (Aurora) sets PALS_RANKID / PALS_LOCAL_SIZE; keep PMI_*/RANK as
    # fallbacks for other launchers. PMI_RANK is EMPTY under PALS -- do not use it.
    for r, s in (("PALS_RANKID", "PALS_LOCAL_SIZE"),
                 ("PMIX_RANK", "PALS_LOCAL_SIZE"),
                 ("PMI_RANK", "PMI_SIZE"),
                 ("RANK", "WORLD_SIZE")):
        if os.environ.get(r):  # present AND non-empty
            return int(os.environ[r]), int(os.environ.get(s) or "1")
    return 0, 1


if __name__ == "__main__":
    dim = int(os.environ.get("DIM", "256"))
    steps = int(os.environ.get("STEPS", "3"))
    device = os.environ.get("DEVICE", "cpu")
    backend = os.environ.get("BACKEND", "gloo")

    rank, world = env_rank()
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    dist.init_process_group(backend=backend, rank=rank, world_size=world)

    torch.manual_seed(rank)
    model = Net(dim).to(device)
    ddp = DDP(model)                       # wraps model; installs grad allreduce hooks
    criterion = nn.MSELoss()
    opt = torch.optim.SGD(ddp.parameters(), lr=0.01)

    for _ in range(steps):
        x = torch.randn(dim, dim, device=device)
        y = torch.randn(dim, dim, device=device)
        opt.zero_grad()
        loss = criterion(ddp(x), y)
        loss.backward()                    # <-- triggers gradient allreduce across ranks
        opt.step()
        if device == "xpu":
            torch.xpu.synchronize()

    print(f"[rank {rank}/{world} pid {os.getpid()}] done "
          f"dim={dim} steps={steps} backend={backend} device={device}", flush=True)
    dist.destroy_process_group()
