#!/usr/bin/env python
"""Profile MyModel's forward+backward pass on CPU with torch.profiler,
exporting a trace TensorBoard's torch_tb_profiler plugin can read.

Run:  source env.sh && python example/profile_cpu.py
"""
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

from model import build

LOGDIR = "../traces/cpu"


def main():
    model, x = build()

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        on_trace_ready=tensorboard_trace_handler(LOGDIR),
    ) as prof:
        output = model(x)
        loss = output.sum()
        loss.backward()

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
    print(f"\nTrace written to {LOGDIR}/ (view with: tensorboard --logdir={LOGDIR})")


if __name__ == "__main__":
    main()
