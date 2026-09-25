# Step 3 — How the granularities change under concurrency

Step 2 measured the RecordFunction trace granularities on a **single-threaded CPU**
run. Two of its results were ties *only because* nothing left the calling thread:
global vs thread-local callbacks were byte-identical (134/134), and the
`thread_local` depth counter was never exercised by more than one thread. This step
breaks that assumption on purpose — real **parallel-training** workloads — and shows
which granularity readings change, and by how much.

Two parallelism axes matter, and RecordFunction treats them very differently:

| Axis | Workload | Threading model | What the hook sees |
|---|---|---|---|
| **Inter-op** (thread-level) | `train_hogwild.py` — Hogwild async SGD, N threads share one model | N OS threads, 1 process | one `vtid` **per thread**, each its own depth stack |
| **Intra-op** (core-level) | `train_intraop.py` — one step, `set_num_threads(N)` | ATen worker pool under one launching thread | **one** `vtid`, work spread across cores *below* the hook |

## Contents

```
03_concurrency_device/
├── README.md                 # this file
├── env.sh                    # module recipe (same as Steps 1–2)
├── example/
│   ├── train_hogwild.py      # inter-op: N threads, shared model, fwd+bwd+step
│   └── train_intraop.py      # intra-op: set_num_threads(N), one big step
├── tracer/
│   ├── tracer.cpp            # unchanged from Step 2 (all toggles)
│   ├── pytorch_tracepoints.tp# LTTng-UST schema: name + scope + depth + args
│   ├── build.sh              # lttng-gen-tp + compile -> libtorch_tracer.so
│   ├── run.sh                # run.sh <label> <workload.py>  (toggles via env)
│   └── view.sh               # babeltrace2 <trace-dir>
└── traces/                   # babeltrace2 dumps
    ├── hogwild_global.txt    # inter-op, global callback   (1954 events, 5 vtid)
    ├── hogwild_local.txt     # inter-op, thread-local      (166 events, 1 vtid)
    └── intraop_global.txt    # intra-op, global callback    (836 events, 1 vtid)
```

## How to reproduce (module-based, no absolute paths)

```bash
source env.sh lttng          # loads oneapi/release/2025.3.1 -> lttng-tools -> babeltrace2 -> frameworks (LAST)
cd tracer && ./build.sh      # -> tracer/libtorch_tracer.so

# inter-op — Hogwild: global catches every thread, thread-local catches one
TRACER_SCOPES=function+backward TRACER_THREAD=global N_THREADS=4 ./run.sh hogwild_global ../example/train_hogwild.py
TRACER_SCOPES=function+backward TRACER_THREAD=local  N_THREADS=4 ./run.sh hogwild_local  ../example/train_hogwild.py

# intra-op — ATen parallel_for under one launching thread
TRACER_SCOPES=function+backward TRACER_THREAD=global N_THREADS=8 DIM=512 ./run.sh intraop_global ../example/train_intraop.py
```

## Where the traces are written

Each experiment lands in its own session, `step3-<label>`. Read one with:

```bash
source env.sh lttng
tracer/view.sh "$(ls -dt $HOME/lttng-traces/step3-hogwild_global-* | head -1)"
# module used to read: babeltrace2/2.1.2-archive (pulled in by `source env.sh lttng`)
```

The `traces/*.txt` files here are those babeltrace2 dumps, saved so the results are
readable without rerunning.

## The results

### Inter-op — the global-vs-thread-local tie finally breaks
Same Hogwild workload (4 worker threads sharing one model, each running
forward → loss → backward → `optimizer.step()`), traced two ways:

```
hogwild_global (addGlobalCallback):      1954 events across 5 vtid
    166  vtid 36736   <- main thread (warm-up + spawn)
    444  vtid 37029   <- worker 1
    448  vtid 37030   <- worker 2
    448  vtid 37031   <- worker 3
    448  vtid 37032   <- worker 4
hogwild_local  (addThreadLocalCallback):  166 events across 1 vtid
    166  vtid 37112   <- main thread ONLY
```

The thread-local callback, installed on the main thread at library load, captured
**166 of 1954 events — it missed 91% of the work**, i.e. every one of the four
training threads. On the single-thread run in Step 2 these two settings were
identical; under real inter-op concurrency they diverge completely. This is the
concrete evidence behind the Step 2 recommendation *always register globally*.

### Depth counter is correctly per-thread
Every one of the 5 vtids has **balanced entry/exit** and its **own** depth stack —
no cross-thread bleed:

```
vtid 36736: entry=83  exit=83  maxdepth=4   BALANCED
vtid 37029: entry=222 exit=222 maxdepth=6   BALANCED
vtid 37030: entry=224 exit=224 maxdepth=6   BALANCED
vtid 37031: entry=224 exit=224 maxdepth=6   BALANCED
vtid 37032: entry=224 exit=224 maxdepth=6   BALANCED
```

`thread_local int g_depth` is the right storage class: each OS thread nests
independently, so depth stays meaningful even when threads interleave in the trace.

### Intra-op parallelism is invisible to the hook
`train_intraop.py` runs a single-threaded (from Python) training loop with
`torch.set_num_threads(8)` on 512×512 matmuls. The work spreads across cores, but
the trace shows **one vtid**:

```
intraop_global:  836 events, 1 vpid, 1 vtid — but work seen on multiple cpu_id
    aten::linear   depth=0   cpu_id=53      <- launched + bracketed on ONE thread
```

RecordFunction brackets the op at `Dispatcher::call` on the **launching** thread;
ATen's `parallel_for` fans the kernel across worker cores *below* that seam, so the
op fires the callback exactly once. **RecordFunction granularity is per-dispatch,
not per-core** — intra-op parallelism does not appear as extra events, and
global-vs-thread-local makes no difference for it.

## What this demonstrates

- **Under real inter-op concurrency, thread-local tracing is not a smaller trace —
  it is a wrong one:** it silently dropped 91% of the work (every training thread).
  Register globally for anything that spawns threads.
- **Global tracing scales per thread:** one balanced, independent depth stack per
  `vtid`, so a multi-threaded trace stays coherent and attributable.
- **Intra-op (core-level) parallelism is invisible here:** ATen `parallel_for` runs
  below the dispatch seam, so a heavily-parallel op is still one event — don't read
  event counts as core utilization.
- **These are two independent axes.** RecordFunction sees inter-op (thread) work but
  not intra-op (core) work; a full picture needs a second signal for the latter.
