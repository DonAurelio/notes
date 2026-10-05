# ITT deep dive: how `torch.autograd.profiler.emit_itt()` actually works

**Goal of this step:** trace `emit_itt()`'s path end to end — Python →
RecordFunction → `pushITTCallbacks` → vendored `ittnotify` → Intel VTune —
with real VTune results and source citations, not just the Python-level
API. Companion to `../kineto-deep-dive/`, which covers the other profiler
backend (`torch.profiler.profile()`).

CPU and XPU are both validated live on this Aurora node, with real VTune
collections for each. **CUDA is out of scope by design**, not deferred —
see "Device scope" below.

## Contents

```
env.sh                   module recipe (oneapi -> frameworks)
example/
  model.py               shared workload (Linear(10,5) -> relu)
  profile_itt.py          wraps forward+backward in emit_itt(); DEVICE=cpu|xpu
results/
  vtune_hotspots/         CPU run: full VTune result dir (~588 MB, not for git)
                           + summary.txt + console_log.txt (small, these travel with the repo)
  xpu_validation/          XPU run (reserved compute node): same, under results/xpu_validation/
docs/
  00-overview.md          layering diagram (ASCII + Mermaid), activation, RecordFunction
                          registration — the SHORT stack: no buffer, no device driver
  01-cpu-path.md           the real VTune "Top Tasks" table, annotated against the source
  02-no-device-support.md  the explicit, evidence-based answer: no CUDA (or XPU-device-
                          activity) path exists — architectural, backed by a live XPU run
```

## How to reproduce

```bash
source env.sh
PYBIN=$(which python3)   # vtune's `--` does not inherit the sourced module PATH

cd example
vtune -collect hotspots -result-dir ../results/vtune_hotspots -- "$PYBIN" profile_itt.py
vtune -report summary -r ../results/vtune_hotspots -format=text -report-knob show-issues=false

# XPU: same script, same command, run on a reserved compute node
DEVICE=xpu vtune -collect hotspots -result-dir ../results/xpu_validation/vtune_hotspots_xpu -- "$PYBIN" profile_itt.py
```

No compute-node reservation is needed for the CPU run — unlike
`torch.profiler.profile()`, `emit_itt()` never touches a device activity
collector, so it runs fine on the Aurora login node (see `docs/00-overview.md`).

## Device scope

ITT is Intel-specific marker tooling paired with VTune. It has:

- **CPU**: fully supported, verified with a real finalized VTune result
  (`docs/01-cpu-path.md`).
- **XPU**: fully supported for RecordFunction-level op markers, verified
  live on a reserved compute node — ranges are identical to the CPU run,
  with no device-specific data (`docs/02-no-device-support.md` §4). ITT has
  no *device-activity* (kernel/runtime) collection on XPU either way — that
  is Kineto/PTI's job, not ITT's.
- **CUDA**: **not supported, by design, not a gap.** There is no device
  branching anywhere in `emit_itt`'s source, zero GPU-related symbols in
  the compiled library, and CUDA's own marker API is a separate, unrelated
  mechanism (`torch.autograd.profiler.emit_nvtx()` / `torch.cuda.nvtx`,
  paired with NVIDIA Nsight/nvprof, not VTune). See
  `docs/02-no-device-support.md` for the full evidence chain. Unlike
  `../kineto-deep-dive/`, there is no `profile_cuda_itt.py` to run later —
  this topic is closed, not deferred.

## What this demonstrates

- `emit_itt()` and `torch.profiler.profile()` share one C++ entry point
  (`_enable_profiler`/`prepareProfiler`) but diverge immediately: Kineto
  populates an activity set and buffers events for export; ITT passes an
  always-empty set and talks directly to a marker library with nothing
  staged in between.
- VTune's "Top Tasks" are literally RecordFunction op names (`aten::addmm`,
  etc.) — the same names every other observer (Kineto, this project's LTTng
  tracer) sees, confirmed directly in a real VTune report.
- Device support is a real, verifiable, binary question — not asserted
  from reading source alone. It was checked with a live run on each device
  class ITT could plausibly support, and answered directly.
