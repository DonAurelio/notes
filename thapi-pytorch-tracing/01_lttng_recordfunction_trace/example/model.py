#!/usr/bin/env python
"""Shared example model used across the Phase 1 dispatcher experiments.

A minimal nn.Module: an affine layer (Linear) followed by relu. Running
`model(x)` triggers a chain of aten operators through the dispatcher, which is
what we trace/observe in the sibling scripts.

Run:  source scripts/env.sh && python examples/model.py
"""
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
    return model, x


if __name__ == "__main__":
    model, x = build()
    print(model)
    output = model(x)
    print("output:", output)
    print("output.grad_fn:", output.grad_fn)
