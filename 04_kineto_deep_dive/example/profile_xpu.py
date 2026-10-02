#!/usr/bin/env python
"""Profile MyModel's forward+backward pass on XPU (Intel GPU) with
torch.profiler, exporting a trace TensorBoard's torch_tb_profiler plugin
can read.

Requires a compute node with XPUs (torch.xpu.device_count() > 0).

Run:  source env.sh && python example/profile_xpu.py
"""
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

from model import build

LOGDIR = "../traces/xpu"


def main():
    assert torch.xpu.is_available(), "no XPU device visible — run on a compute node"
    device = torch.device("xpu")

    model, x = build()
    model = model.to(device)
    x = x.to(device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        on_trace_ready=tensorboard_trace_handler(LOGDIR),
    ) as prof:
        output = model(x)
        loss = output.sum()
        loss.backward()
        torch.xpu.synchronize()

    print(prof.key_averages().table(sort_by="self_xpu_time_total", row_limit=15))
    print(f"\nTrace written to {LOGDIR}/ (view with: tensorboard --logdir={LOGDIR})")


if __name__ == "__main__":
    main()
