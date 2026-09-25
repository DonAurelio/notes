#!/usr/bin/env python
"""Intra-op parallel training step — simplified real workload (Step 4a).

A SINGLE-threaded (from Python's view) training loop, but with ATen intra-op
parallelism turned up via torch.set_num_threads(N). One large matmul-heavy step
lets ATen fan a single op (e.g. addmm) across its worker-thread pool.

Why this workload for the concurrency experiment:
  * The Python code launches ops from ONE thread, but ATen may execute a single
    op's kernel across N worker threads internally.
  * Open question this answers: does RecordFunction fire on those ATen worker
    threads, or only on the launching thread? RecordFunction brackets the op at
    Dispatcher::call on the LAUNCHING thread -- the parallel_for inside the
    kernel is below that seam -- so we expect ONE vtid, not N, even though N
    cores do the work. This makes the point that RecordFunction granularity is
    per-DISPATCH, not per-core.

Env:
  N_THREADS   ATen intra-op threads (default 8)
  DIM         matmul dimension, bigger => more parallel work (default 512)
  STEPS       training steps (default 3)

Run:  source scripts/env.sh && python examples/train_intraop.py
"""
import os

import torch
import torch.nn as nn


class BigModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim)

    def forward(self, x):
        return torch.relu(self.l2(torch.relu(self.l1(x))))


if __name__ == "__main__":
    n = int(os.environ.get("N_THREADS", "8"))
    dim = int(os.environ.get("DIM", "512"))
    steps = int(os.environ.get("STEPS", "3"))

    torch.set_num_threads(n)  # ATen intra-op pool size
    torch.manual_seed(0)

    model = BigModel(dim)
    criterion = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    for _ in range(steps):
        x = torch.randn(dim, dim)
        y = torch.randn(dim, dim)
        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()

    print(f"done: intra-op {n} threads, dim={dim}, {steps} steps")
