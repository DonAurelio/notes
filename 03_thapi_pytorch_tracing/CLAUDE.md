# CLAUDE.md — THAPI pytorch-backend validation

> **This repo is documentation only.** All experiments — building THAPI,
> recording traces, running workloads — are conducted in a separate working
> directory, NOT here. Only final notes and write-ups are committed to this repo.
> Never scaffold, build, or record inside it.

## What this project is

Unlike [02_pytorch_tracing](../02_pytorch_tracing/), which builds a **standalone**
RecordFunction tracer to understand the dispatcher hook mechanism itself, this
stage validates [THAPI](https://github.com/argonne-lcf/THAPI)'s own shipped
`pytorch` backend (`libTracerPytorch.so` + `iprof` orchestration) across device
(CPU/XPU), multi-process (MPI), and multi-node axes. The question here is
whether THAPI's own pipeline is correct under these conditions, not whether
RecordFunction works (that's 02_pytorch_tracing's territory).

## Layout

```
example/model.py       # shared workload: Linear(10,5) -> relu -> sum -> backward, DEVICE-toggled
analyze_thapi.py        # THAPI-raw-trace-schema analyzer (entry/exit balance, vtid/rank, backward-thread)
env.sh                  # module recipe + THAPI_INSTALL path hookup
00_cpu_baseline/         # control: 1 process, CPU
01_xpu_single/           # 1 process, XPU — device autograd thread appears
02_mpi_cpu/              # 2 ranks, co-located, CPU
03_mpi_xpu/              # 2 ranks, co-located, XPU
04_mpi_2node_cpu/        # 2 ranks, 1 per node, CPU
05_mpi_2node_xpu/        # 2 ranks, 1 per node, XPU
```

Each `NN_*` experiment holds only `traces/` (raw/interval/tally text + a
`.pftrace` timeline) — no per-experiment tracer or example dir, since all six
runs share the one `example/model.py` and THAPI's already-built `iprof`. The
top-level `README.md` carries the toggle table, reproduce recipe, and results
for all six experiments together.

## Conventions

- **Documentation only** — run every experiment elsewhere; commit only results.
- **Results cite numbers, not adjectives** — every finding in the README is
  backed by a fenced excerpt of `analyze_thapi.py` / `iprof` tally output.
- THAPI's raw-trace text schema (`vtid: N`, unquoted `name: ...`) differs from
  02_pytorch_tracing's hand-rolled tracer schema (`vtid = N`, quoted names) —
  `analyze_thapi.py` is schema-specific; don't reuse 02_pytorch_tracing's
  analysis tooling on these traces, it silently mis-parses them.

## Gotchas that will bite (not in the README)

- **`iprof` wraps each MPI rank — it does not wrap `mpirun`.** `mpirun -n 2 --
  iprof ...` is correct; `iprof -- mpirun ...` is explicitly the wrong direction
  (`iprof` itself warns on this) and only profiles a single hostname.
- **MPI needs a PBS-allocated compute node even for CPU-only ranks** — `mpirun -n
  2 hostname` exits 127 on the Aurora login node.
- **Multi-node recording: point `--trace-output` at a path under the shared
  `$HOME` filesystem**, not node-local `/tmp` — otherwise each node's `iprof`
  writes its half of the trace somewhere the other node/head can't see.
- **`--trace-output`'s metadata type gets mutated by whatever `iprof` mode runs
  first.** Recording with the default (tally) mode leaves the trace dir marked
  `:type: aggreg`, which then refuses to replay as raw/timeline. Always record
  with `-t` (raw) first if you need multiple views from one recorded session,
  then replay the others with `-r`.
- **`babeltrace_thapi trace ... --restrict` silently produces empty output**
  against these traces — drop `--restrict`, keep `--context --backends
  pytorch:6[,mpi:3]`.
- **`babeltrace_thapi trace`/`to_interval` take one positional trace-dir argument
  per hostname** — for a multi-node trace, list every `<dir>/<hostname>` subdir
  explicitly; there's no automatic "read the whole `--trace-output` tree" mode.
- Module env does NOT propagate across shells or to compute nodes — re-`source
  env.sh` everywhere; load `frameworks` LAST or `import torch` fails on a
  `sycl::queue` symbol (same gotcha as 02_pytorch_tracing).

## Open directions

- Larger rank counts (4+, matching 02_pytorch_tracing's DDP stages) to see if
  THAPI's per-rank aggregation still holds at scale.
- A real DDP/collective workload (this stage used the same tiny non-distributed
  model on every rank, not `torch.distributed` — no `c10d::allreduce_` traffic
  to observe here, unlike 02_pytorch_tracing's Steps 5–6).
- Device-timing accuracy: THAPI's interval `dur` field vs. actual device
  execution time on XPU (host-launch-vs-device-exec, the same question
  02_pytorch_tracing's Step 4 raised for the hand-written tracer).
