#!/usr/bin/env python
"""Hogwild async SGD — simplified real CPU parallel-training workload (Step 4a).

N Python threads share ONE model and each runs a full training step
(forward -> loss -> backward -> optimizer.step()) in a loop. This is the classic
Hogwild! lock-free async SGD pattern: one process, N OS threads, shared params.

Why this workload for the concurrency experiment:
  * Each worker is its own OS thread => its own vtid in the trace.
  * A GLOBAL RecordFunction callback catches every worker's ops; a THREAD-LOCAL
    callback (installed on the main thread at library load) sees NONE of them.
    On the single-thread CPU run in Step 3 these two were identical (134/134) --
    here they must DIVERGE. That divergence is the thing we are measuring.
  * Each vtid carries its own thread_local depth stack, so this also tests that
    the tracer's depth counter is correctly per-thread.

Env:
  N_THREADS   number of worker threads (default 4)
  STEPS       training steps per worker (default 3)

Run:  source scripts/env.sh && python examples/train_hogwild.py
"""
import os
import threading

import torch
import torch.nn as nn

# Pin intra-op parallelism to 1 so the ONLY threads are our workers -- keeps the
# vtid set clean and attributable to Hogwild threads (contrast: train_intraop.py).
torch.set_num_threads(1)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return torch.relu(self.linear(x))


def train_worker(model, criterion, steps):
    # Fresh optimizer per thread, shared params (Hogwild): updates race by design.
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(steps):
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()


if __name__ == "__main__":
    torch.manual_seed(0)
    n = int(os.environ.get("N_THREADS", "4"))
    steps = int(os.environ.get("STEPS", "3"))

    model = MyModel()
    model.share_memory()  # params shared across threads (Hogwild)
    criterion = nn.MSELoss()

    # Warm up on the main thread so one-time lazy-init ops are attributed here,
    # not to a worker vtid.
    _w = torch.randn(4, 10)
    model(_w).sum().backward()
    model.zero_grad()

    threads = [
        threading.Thread(target=train_worker, args=(model, criterion, steps))
        for _ in range(n)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"done: hogwild {n} threads x {steps} steps")
