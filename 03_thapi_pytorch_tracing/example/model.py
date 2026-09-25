#!/usr/bin/env python
"""Shared example model used across the THAPI pytorch-backend validation.

A minimal nn.Module: an affine layer (Linear) followed by relu. Running
`model(x)` triggers a chain of aten operators through the dispatcher, which is
what we trace/observe with iprof's pytorch backend.

DEVICE env var (default "cpu") selects the target device. When DEVICE=xpu,
PALS_LOCAL_RANKID picks the local tile so co-located MPI ranks don't collide
on device 0.

Run:  source env.sh && python3 model.py
"""
import os

import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        """Define the layers."""
        super().__init__()
        # an affine operation: y = Wx + b
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        """Connect the layers."""
        return torch.relu(self.linear(x))


def build():
    torch.manual_seed(0)
    model = MyModel()
    x = torch.randn(1, 10)

    device = os.environ.get("DEVICE", "cpu")
    if device == "xpu":
        local_rank = int(os.environ.get("PALS_LOCAL_RANKID", "0"))
        device = f"xpu:{local_rank}"
    model = model.to(device)
    x = x.to(device)

    return model, x


if __name__ == "__main__":
    model, x = build()
    print(model)
    output = model(x)
    print("output:", output)
    print("output.grad_fn:", output.grad_fn)

    # Loss: sum of all output elements. This is a stand-in for a real
    # loss function — it just gives us a single scalar to call
    # .backward() on. In a real training loop this would instead be
    # something like nn.MSELoss()(output, target) or
    # nn.CrossEntropyLoss()(output, labels).
    loss = output.sum()
    print("loss:", loss)
    print("loss.grad_fn:", loss.grad_fn)

    # Backward: walks the autograd graph from `loss` back to every
    # leaf tensor that requires grad (here, model.linear.weight and
    # model.linear.bias), accumulating d(loss)/d(param) into each
    # tensor's .grad attribute. This is what actually triggers the
    # BACKWARD_FUNCTION-scoped ops your tracer is watching for.
    loss.backward()

    print("linear.weight.grad:", model.linear.weight.grad)
    print("linear.bias.grad:", model.linear.bias.grad)
