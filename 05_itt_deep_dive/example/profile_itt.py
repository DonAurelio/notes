#!/usr/bin/env python
"""Wrap MyModel's forward+backward pass in torch.autograd.profiler.emit_itt(),
meant to be run under Intel VTune so the ITT ranges show up as VTune tasks.

Unlike torch.profiler.profile() (the Kineto path, see ../../kineto-deep-dive),
emit_itt() does not export a trace file on its own -- it only emits live ITT
markers for an external collector (VTune) that must already be attached.
Running this script directly (without vtune) just exercises the code path
with no visible effect.

Run under VTune (use the explicit interpreter path -- vtune's `--` does NOT
inherit the sourced module PATH the way a plain shell does):

  source env.sh
  PYBIN=$(which python3)
  cd example
  vtune -collect hotspots -result-dir ../results/vtune_hotspots -- "$PYBIN" profile_itt.py

To run on XPU instead of CPU, set DEVICE=xpu (used by the device-support
validation in docs/02-no-device-support.md):

  DEVICE=xpu vtune -collect hotspots -result-dir ... -- "$PYBIN" profile_itt.py
"""
import os

import torch
from torch.autograd.profiler import emit_itt
import torch.profiler.itt as itt

from model import build

ITERS = 30000


def main():
    device = os.environ.get("DEVICE", "cpu")
    print("itt.is_available():", itt.is_available())

    model, x = build()
    if device != "cpu":
        model = model.to(device)
        x = x.to(device)

    with emit_itt(record_shapes=True):
        for _ in range(ITERS):
            output = model(x)
            loss = output.sum()
            loss.backward()
            model.zero_grad()
            if device != "cpu":
                getattr(torch, device).synchronize()

    print("emit_itt run complete, device:", device)


if __name__ == "__main__":
    main()
