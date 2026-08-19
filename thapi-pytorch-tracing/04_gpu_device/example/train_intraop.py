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
  DEVICE      cpu (default) | xpu -- run the ops on that device. On xpu the
              intra-op thread pool is irrelevant to the kernel (it runs on the
              device), but RecordFunction still brackets the HOST-side launch;
              this is the step-3b/3c device run.
  SYNC        0 (default) | 1 -- whether the workload calls torch.xpu.synchronize()
              at the end of each step. This is deliberately a TOGGLE, not baked in:
              real user code may or may not synchronize, and it changes what the
              RecordFunction entry/exit interval means on an async device --
              without sync the op returns after the host-side launch (kernel still
              running on device); with sync the trailing sync op's interval absorbs
              the device wait. We want to be able to trace BOTH. No effect on cpu.

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
    device = os.environ.get("DEVICE", "cpu")
    do_sync = os.environ.get("SYNC", "0") == "1"

    torch.set_num_threads(n)  # ATen intra-op pool size (host-side; cpu-relevant)
    torch.manual_seed(0)

    model = BigModel(dim).to(device)
    criterion = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    for _ in range(steps):
        x = torch.randn(dim, dim, device=device)
        y = torch.randn(dim, dim, device=device)
        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        # SYNC toggle: only meaningful on an async device (xpu). Left OFF by
        # default so the default trace shows host-launch semantics; flip SYNC=1
        # to fold the device wait into a trailing synchronize op.
        if do_sync and device == "xpu":
            torch.xpu.synchronize()

    print(f"done: intra-op {n} threads, dim={dim}, {steps} steps, "
          f"device={device}, sync={do_sync}")
