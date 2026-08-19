#!/usr/bin/env python
"""Forward + backward workload for the granularity experiments (Step 3).

Same MyModel as examples/model.py, but we run a backward pass so the trace
contains autograd nodes (BACKWARD_FUNCTION scope) — which run on a separate
autograd thread. This is what makes the scope (aspect 2) and thread (aspect 3)
granularities observable.

Run:  source scripts/env.sh && python examples/model_backward.py
"""
import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return torch.relu(self.linear(x))


if __name__ == "__main__":
    torch.manual_seed(0)
    model = MyModel()
    x = torch.randn(1, 10)
    output = model(x)
    loss = output.sum()
    loss.backward()          # <- triggers the autograd backward pass
    print("loss:", loss.item())
    print("linear.weight.grad is set:", model.linear.weight.grad is not None)
