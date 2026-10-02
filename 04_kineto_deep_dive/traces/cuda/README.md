# CUDA trace — not yet recorded

This system (Aurora) has no NVIDIA/CUDA hardware, so `example/profile_cuda.py`
has not been run here. `docs/04-cuda-path.md` documents the expected CUDA
path from upstream `pytorch/kineto` source only, and is explicitly flagged
as unverified against a real trace.

To complete this:

1. On a CUDA-capable system, with a torch build that has CUDA/CUPTI support:
   ```bash
   source env.sh   # or your local equivalent module/env setup
   cd example
   python3 profile_cuda.py
   ```
2. Copy the resulting `traces/cuda/*.pt.trace.json` into this directory.
3. Re-check `docs/04-cuda-path.md`'s predictions (event categories, field
   names, which kernel/runtime names appear) against the real trace, and
   correct anything that doesn't match.
