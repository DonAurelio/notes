# Step 4 — Kineto deep dive: how `torch.profiler.profile()` actually works

**Goal of this step:** trace `torch.profiler.profile()`'s Kineto backend end to end —
Python → RecordFunction → Kineto → PTI (XPU) / CUPTI (CUDA) → driver → back up as a
stored event — with real traces and source citations, not just the Python-level API.

CPU and XPU are fully recorded and verified on this Aurora node. **CUDA is not** — no
NVIDIA hardware here — see "What's missing" below.

## Contents

```
env.sh                   module recipe (oneapi -> frameworks)
example/
  model.py               shared workload (Linear(10,5) -> relu)
  profile_cpu.py          torch.profiler run, CPU only   -> traces/cpu/
  profile_xpu.py          torch.profiler run, CPU+XPU     -> traces/xpu/
  profile_cuda.py         torch.profiler run, CPU+CUDA    -> traces/cuda/  (NOT YET RUN)
traces/
  cpu/*.pt.trace.json     recorded
  xpu/*.pt.trace.json     recorded
  cuda/README.md          stub only — no trace yet, see below
docs/
  00-overview.md          full layering diagram (ASCII + Mermaid), activation, collection
  01-event-model.md       event classification, storage, same-vs-different per activity
  02-cpu-path.md          real aten::addmm + real memory event, annotated field-by-field
  03-xpu-path.md          real gemm_kernel event, PTI structs, full correlation chain
  04-cuda-path.md         CUDA analogue from upstream kineto source only — UNVERIFIED
```

## How to reproduce (CPU / XPU)

```bash
source env.sh
pip install --user tensorboard torch_tb_profiler   # one-time

# torch.profiler needs a compute node, even for CPU-only (see docs/00-overview.md)
cd example
python3 profile_cpu.py
python3 profile_xpu.py
```

## What's missing: CUDA

`docs/04-cuda-path.md` is written entirely from reading `pytorch/kineto`'s upstream
source (at the exact commit this PyTorch build pins) — **no CUDA code here has been
compiled or run**. To close this out on a CUDA-capable system:

1. `example/profile_cuda.py` is already written (same shape as `profile_xpu.py`) —
   just run it: `python3 profile_cuda.py`.
2. Copy the resulting `traces/cuda/*.pt.trace.json` into `traces/cuda/` here.
3. Check `docs/04-cuda-path.md` §4's predictions (event category names, the
   `cpu_op` → `cuda_runtime` → `kernel` correlation chain) against the real trace,
   and correct anything that doesn't match — §6 of that doc has the full checklist.

## What this demonstrates

- `torch.profiler` needs a real device context even to profile CPU-only code — it
  fails on Aurora's login node regardless of requested activities.
- RecordFunction is the one hook every backend (Kineto, ITT, this project's own
  LTTng tracer) attaches to; Kineto just merges that CPU-side stream with a
  vendor-specific device-activity stream (PTI on XPU, CUPTI on CUDA).
- The event taxonomy and stored shape are identical across CPU/XPU/CUDA — only the
  vendor wire protocol feeding into it differs.
