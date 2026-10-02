#!/usr/bin/env python
"""Profile MyModel's forward+backward pass on CUDA (NVIDIA GPU) with
torch.profiler, exporting a trace TensorBoard's torch_tb_profiler plugin
can read.

NOT EXECUTED on Aurora — this system has no NVIDIA/CUDA hardware
(torch.cuda.is_available() is always False here). This script is modeled
directly on profile_xpu.py (same activities/record_shapes/with_stack/
profile_memory/on_trace_ready shape, swapping "xpu" for "cuda") and is meant
to be run on a CUDA-capable system to complete the CUDA findings in
docs/04-cuda-path.md.

Requires a CUDA-capable node (torch.cuda.is_available() == True) with a
torch build that has CUDA/CUPTI support compiled in.

Run:  source env.sh && python example/profile_cuda.py
After running, copy the resulting traces/cuda/*.pt.trace.json back here so
docs/04-cuda-path.md's predictions can be checked against a real trace.
"""
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

from model import build

LOGDIR = "../traces/cuda"


def main():
    assert torch.cuda.is_available(), "no CUDA device visible — run on a CUDA-capable node"
    device = torch.device("cuda")

    model, x = build()
    model = model.to(device)
    x = x.to(device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        on_trace_ready=tensorboard_trace_handler(LOGDIR),
    ) as prof:
        output = model(x)
        loss = output.sum()
        loss.backward()
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
    print(f"\nTrace written to {LOGDIR}/ (view with: tensorboard --logdir={LOGDIR})")


if __name__ == "__main__":
    main()
