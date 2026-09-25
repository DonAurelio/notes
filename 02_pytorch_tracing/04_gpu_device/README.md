# Step 4 — How the granularities change on GPU (XPU)

Step 3 measured the RecordFunction trace granularities under **CPU concurrency**.
One of its headline results was that on CPU, backward runs *inline on the calling
thread*, so the granularities never separated purely because of the device. This
step re-runs the same two workloads on an **Aurora GPU (XPU)** and shows what the
device itself changes — a separate autograd thread, host-launch-vs-device-exec
timing, and `@xpu` in the args.

Nothing in the tracer changed from Step 3: it is the **same `.so`**. Device and
synchronization are load-time toggles on the same workloads:

| Toggle | Values | Effect |
|---|---|---|
| `DEVICE` | `cpu` \| `xpu` | run the ops on CPU or on an Aurora GPU |
| `SYNC` | `0` (default) \| `1` | whether the workload calls `torch.xpu.synchronize()` each step |

`SYNC` is a deliberate axis, not a fixed choice: real user code may or may not
synchronize, and it changes what a RecordFunction entry/exit interval *means* on an
async device (see "Host-launch vs device-exec" below).

These runs require a real device — the login/head node reports
`XPU device count is zero`. All traces here were captured on an Aurora **compute
node** (12 XPUs), reached with `qsub -A Performance -q debug -l walltime=00:60:00
-l filesystems=home -l select=1 -I`; home is shared, so the code and `.so` are
already visible on the node.

## Contents

```
04_gpu_device/
├── README.md                    # this file
├── env.sh                       # module recipe (same as Steps 1–3)
├── example/
│   ├── train_hogwild.py         # inter-op: N threads share one model (DEVICE/SYNC aware)
│   └── train_intraop.py         # intra-op: one big step            (DEVICE/SYNC aware)
├── tracer/
│   ├── tracer.cpp               # UNCHANGED from Step 3 — renders @device already
│   ├── pytorch_tracepoints.tp   # LTTng-UST schema: name + scope + depth + args
│   ├── build.sh                 # lttng-gen-tp + compile -> libtorch_tracer.so
│   ├── run.sh                   # run.sh <label> <workload.py>  (toggles via env)
│   └── view.sh                  # babeltrace2 <trace-dir>
└── traces/                      # babeltrace2 dumps (compute node, 12×XPU)
    ├── xpu_intraop_nosync.txt   # intra-op, xpu, SYNC=0     (400 events, 2 vtid)
    ├── xpu_intraop_sync.txt     # intra-op, xpu, SYNC=1     (identical op multiset)
    ├── xpu_inputs.txt           # intra-op, xpu, INPUTS=1   (@xpu:0 in args)
    ├── xpu_hogwild_global.txt   # inter-op, xpu, global     (891 events, 6 vtid)
    └── xpu_hogwild_local.txt    # inter-op, xpu, thread-local (74 events, 2 vtid)
```

## How to reproduce (compute node)

```bash
# on a compute node, home is shared so the code + .so are already visible:
source env.sh lttng          # oneapi/release/2025.3.1 -> lttng -> babeltrace2 -> frameworks (LAST)
export LTTNG_HOME="/tmp/lttng_${USER}_$$"  # shared $HOME/.lttng sessiond lock blocks the node; isolate it
cd tracer && ./build.sh      # -> tracer/libtorch_tracer.so  (same source as Step 3)

# 4a intra-op on device — host-launch semantics vs a trailing synchronize
TRACER_SCOPES=function+backward DEVICE=xpu SYNC=0 DIM=512 ./run.sh xpu_intraop_nosync ../example/train_intraop.py
TRACER_SCOPES=function+backward DEVICE=xpu SYNC=1 DIM=512 ./run.sh xpu_intraop_sync   ../example/train_intraop.py
# device shows up in the args (aspect 5 under device)
TRACER_SCOPES=function+backward TRACER_INPUTS=1 DEVICE=xpu DIM=256 ./run.sh xpu_inputs ../example/train_intraop.py

# 4b multi-thread + device — global catches the autograd thread, thread-local does not
TRACER_SCOPES=function+backward TRACER_THREAD=global DEVICE=xpu N_THREADS=4 ./run.sh xpu_hogwild_global ../example/train_hogwild.py
TRACER_SCOPES=function+backward TRACER_THREAD=local  DEVICE=xpu N_THREADS=4 ./run.sh xpu_hogwild_local  ../example/train_hogwild.py
```

## Where the traces are written

With `LTTNG_HOME` set (required on the compute node), each experiment lands in
`$LTTNG_HOME/lttng-traces/step3-<label>-<timestamp>/`. Read one with:

```bash
source env.sh lttng
tracer/view.sh "$(ls -dt $LTTNG_HOME/lttng-traces/step3-xpu_intraop_nosync-* | head -1)"
```

The `traces/*.txt` files here are those babeltrace2 dumps, saved so the results are
readable without a device.

## The results

### The CPU tie breaks: backward runs on its own thread
On CPU (Step 2), global and thread-local callbacks were identical because the
autograd backward pass ran **inline on the calling thread**. On XPU that is no
longer true. The single-threaded (from Python) `train_intraop.py` produces **two
vtids**:

```
xpu_intraop_nosync:  400 events across 2 vtid
    vtid 69957  entry=181  exit=181  maxdepth=3   <- main thread: FORWARD ops
    vtid 70050  entry=219  exit=219  maxdepth=5   <- autograd engine thread: BACKWARD
```

Every backward event — 132 of them, all `*Backward0` / `AccumulateGrad` /
`autograd::engine::evaluate_function:*` — lands on vtid 70050; the main thread has
**zero** backward events. The autograd engine dispatches the backward graph on a
**dedicated worker thread** on this device. That is the single most important
device effect: the global-vs-thread-local decision now matters even for a program
that never spawned a thread itself.

### Multi-thread + device: global sees the autograd thread, thread-local misses everything
Hogwild (4 worker threads sharing one model) on XPU:

```
xpu_hogwild_global (addGlobalCallback):    891 events across 6 vtid
    vtid 71795   82   <- main thread (warm-up + spawn)
    vtid 71880  942*  <- autograd engine thread (312 backward events, maxdepth 5)
    vtid 71892  192   <- worker 1 (forward only, maxdepth 3)
    vtid 71893  192   <- worker 2
    vtid 71894  192   <- worker 3
    vtid 71895  182   <- worker 4
xpu_hogwild_local  (addThreadLocalCallback):  74 events across 2 vtid
    -> captured 74 / 891 = 8.3%   (missed 91.7%)
                              (* line count in the dump, not entry count)
```

The device adds a **sixth thread** the CPU run never had — the autograd engine —
and it carries *more* events than any single worker (all four workers' backward
work is funneled through it). A thread-local callback installed on the main thread
sees only 8.3% of the trace. **On device, "register globally" is not a
nice-to-have; without it the entire backward graph is invisible.** Every vtid still
has balanced entry/exit and its own depth stack — the `thread_local` depth counter
holds up with a device runtime in the mix.

### Host-launch vs device-exec: `synchronize()` is invisible to the hook
Running `train_intraop.py` with `SYNC=0` vs `SYNC=1` (the workload calls
`torch.xpu.synchronize()` after each step) produces **byte-identical op multisets**:

```
diff(op-names nosync, op-names sync)  ->  (empty): SYNC adds no traced event
grep synchronize xpu_intraop_sync.txt ->  0 matches
```

`torch.xpu.synchronize()` is **not a dispatched ATen op**, so RecordFunction never
brackets it. The consequence is a semantic gap CPU tracing never showed: a device
op's entry→exit interval is the **host-side launch** time, *not* the kernel's GPU
time — the kernel runs asynchronously below the seam. Whether or not user code
synchronizes does not change the events; it only changes *when the host actually
waits*, and that wait is invisible here. **This hook measures launch structure, not
device time** — kernel timing needs a device-side profiler (Level-Zero/Kineto).

### Device shows up in the args for free
With `TRACER_INPUTS=1`, tensors render their device — no tracer change was needed,
because it already emits `tensor.device().str()`:

```
aten::addmm   args="Float[256]@xpu:0, Float[256,256]@xpu:0, Float[256,256]@xpu:0, scalar, scalar"
aten::copy_   args="Float[256,256]@xpu:0, Float[256,256]@cpu, scalar"   <- host->device staging visible
```

345 of the argument tensors render `@xpu:0`; the `@cpu` ones are exactly the
host→device `copy_` / staging tensors, so the trace also makes data movement
visible (aspect 5 under device).

## What this demonstrates

- **On GPU the autograd backward pass runs on a dedicated engine thread,** not
  inline — so global-vs-thread-local diverges even for single-threaded Python code.
  On CPU this difference was hidden.
- **"Register globally" is mandatory on device:** a thread-local callback saw 8.3%
  of a Hogwild-on-XPU trace and missed the entire backward graph.
- **The entry/exit interval is host-launch time, not kernel time.** `synchronize()`
  is not a dispatched op and leaves no event; this hook captures launch *structure*,
  and device *timing* needs a separate device-side signal.
- **Device is captured for free** in `@device` args — including host↔device `copy_`,
  which makes data staging visible with no extra instrumentation.
- **The design holds under a device runtime:** every vtid (workers + main + autograd
  engine) has balanced entry/exit and an independent `thread_local` depth stack.
