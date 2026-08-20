# CLAUDE.md — THAPI PyTorch tracing research

## What this project is

Understand the PyTorch **dispatcher ↔ RecordFunction** mechanism by observing real
op flow — NOT by instrumenting Python:

- **RecordFunction** = the *hook*: the observer callback PyTorch invokes on every
  dispatched op, from inside `Dispatcher::call`.
- **`LD_PRELOAD`** = the *delivery*: a C++ tracer built as `libtorch_tracer.so`,
  injected into an **unmodified** `model.py`.
- **LTTng-UST** = the *backend*: stamps time + `vpid`/`vtid`, writes CTF; read with
  `babeltrace2`.

One `.so`, **build-once / configure-at-runtime**: behaviour is selected by
`TRACER_*` env vars, never by recompiling.

## Layout

```
00_pytorch_introduction/   # pre-tracing background notes (PyTorch, architecture, ML) + img/
01_lttng_recordfunction_trace/   # Step 1 — prove the hook: preload → LTTng CTF
02_recordfunction_granularities/ # Step 2 — the 6 granularity toggles, single-threaded CPU
03_concurrency_device/           # Step 3 — granularities under CPU concurrency
04_gpu_device/                   # Step 4 — same workloads on Aurora XPU
```

Each `NN_*` stage is self-contained (`README.md`, `env.sh`, `example/`, `tracer/`,
`traces/`) and its README carries the toggle table, reproduce recipe, and results.
**Start from the stage README you're extending** — this file only orients.

## Conventions

- **README is public; RUNBOOK is local-only** — never commit a `RUNBOOK.md` here.
- **Results cite numbers, not adjectives** (per-vtid entry/exit, maxdepth, event
  totals, capture %) — get them from the analyze tooling, don't eyeball.
- Stage README shape: title → intro → toggle table → contents tree (annotate each
  trace `(N events, M vtid)`) → reproduce → `### finding` per result → "What this
  demonstrates".
- `00_pytorch_introduction/` is background reading, not a tracing stage.

## Gotchas that will bite (not in the stage READMEs)

- **Module env does NOT propagate** across shells or to compute nodes — re-`source
  env.sh` everywhere; load `frameworks` **LAST** or `import torch` fails on a
  `sycl::queue` symbol.
- **Build the `.so` with `-DNDEBUG -D_GLIBCXX_USE_CXX11_ABI=1`** or RecordFunction's
  ABI breaks (`fn.inputs()` assert / entry-exit desync).
- **GPU needs a compute node** (head node has 0 XPU); there, isolate LTTng with
  `export LTTNG_HOME="/tmp/lttng_${USER}_$$"` (shared `$HOME/.lttng` lock).

## Open directions

- **Per-dispatch-key granularity is out of reach for RecordFunction** (redispatch
  is uninstrumented; no DispatchKey accessor). Reaching it needs a different
  mechanism — `TorchDispatchMode` or raw-symbol interposition of
  `KernelFunction::call`/`lookup`. Natural next stage.
- Multi-device / multi-process (multiple XPUs, collectives) tracing.
- Overhead quantification vs a clean baseline.
