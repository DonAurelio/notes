# Step 6 — How the granularities behave across MPI processes (DDP), GPU

[Step 5](../05_mpi_cpu/README.md) ran `DistributedDataParallel` (DDP) across MPI
ranks on **CPU** (gloo) and showed the process axis is per-`vpid` coherent for
free. It closed on a *tie*: because CPU backward runs inline on each rank's main
thread, a thread-local callback lost nothing — exactly the Step 2 tie, reappearing
per-rank. This step runs the **same DDP workload on the GPU** — one XPU tile per
rank, gradients synchronized with torch's native `xccl` backend — to see the tie
break: on device the autograd engine gets its **own thread per rank**, the same
effect [Step 4](../04_gpu_device/README.md) found single-process, now multiplied
across ranks.

The tracer is **UNCHANGED — the same `.so` as Steps 4–5.** GPU is a *device* axis
and MPI is a *launch* axis; neither is a tracer change. LTTng's `vpid` context
separates the ranks and `vtid` separates each rank's threads, exactly as before.

The backend is torch 2.10's **native `xccl`** (XPU collectives, built into
`torch.distributed`) — the external `oneccl_bindings_for_pytorch` module is neither
installed nor needed. Each rank binds to its own tile (`xpu:{local_rank}`) before
`init_process_group`, so rank *r* runs on `xpu:r`.

| Toggle | Values | Effect |
|---|---|---|
| `NRANKS` | `1` \| `4` (default) | number of MPI ranks `mpiexec` spawns (processes = `vpid`s) |
| `DEVICE` | `xpu` | run replicas on the GPU (one tile per rank) |
| `BACKEND` | `xccl` (xpu default) \| `gloo` \| `mpi` | `torch.distributed` collective backend; `xccl` = native XPU |
| `TRACER_THREAD` | `global` (default) \| `local` | register callback on every thread vs the load thread only |
| `TRACER_INPUTS` | `0` (default) \| `1` | render `fn.inputs()` — exposes device + collective tensor args |

`DEVICE=xpu`/`BACKEND=xccl` are the new axes; the rest carry over from Steps 2–5. A
compute node is required (the login node has zero XPUs).

## Contents

```
06_mpi_gpu/
├── README.md                    # this file
├── env.sh                       # module recipe (same across stages)
├── example/
│   └── train_ddp.py             # DDP: N ranks, one replica each, allreduce every backward
├── tracer/
│   ├── tracer.cpp               # UNCHANGED from Steps 4–5 — same .so
│   ├── pytorch_tracepoints.tp   # LTTng-UST schema: name + scope + depth + args
│   ├── build.sh                 # lttng-gen-tp + compile -> libtorch_tracer.so
│   ├── run.sh                   # single-process recorder (Steps 1–4)
│   ├── run_mpi.sh               # mpiexec -n N into ONE shared LTTng session
│   └── view.sh                  # babeltrace2 <trace-dir>
└── traces/                      # babeltrace2 dumps
    ├── ddp_xpu_n1.txt           # baseline, 1 rank, global          (1096 events, 2 vtid, 1 vpid)
    ├── ddp_xpu_n4_global.txt    # 4 ranks, global callback          (4624 events, 8 vtid, 4 vpid)
    ├── ddp_xpu_n4_local.txt     # 4 ranks, thread-local callback    (4624 events, 8 vtid, 4 vpid)
    └── ddp_xpu_n4_inputs.txt    # 4 ranks, global, INPUTS=1         (4624 events, 8 vtid, device args)
```

## How to reproduce

```bash
# On an Aurora compute node (12 XPUs; the login node has none).
source env.sh lttng                       # oneapi -> lttng -> babeltrace2 -> frameworks (LAST)
export LTTNG_HOME="/tmp/lttng_${USER}_$$"  # node-local; avoids shared-home sessiond lock
cd tracer && ./build.sh                    # -> tracer/libtorch_tracer.so (same source as Steps 4-5)

# baseline: single XPU rank
NRANKS=1 TRACER_SCOPES=function+backward TRACER_THREAD=global BACKEND=xccl DEVICE=xpu \
  ./run_mpi.sh ddp_xpu_n1 ../example/train_ddp.py

# focus: 4 ranks, one tile each, into ONE shared session -> expect 4 vpids
NRANKS=4 TRACER_SCOPES=function+backward TRACER_THREAD=global BACKEND=xccl DEVICE=xpu \
  ./run_mpi.sh ddp_xpu_n4_global ../example/train_ddp.py

# thread-local variant, and device + collective tensor args
NRANKS=4 TRACER_SCOPES=function+backward TRACER_THREAD=local  BACKEND=xccl DEVICE=xpu \
  ./run_mpi.sh ddp_xpu_n4_local ../example/train_ddp.py
NRANKS=4 TRACER_SCOPES=function+backward TRACER_INPUTS=1      BACKEND=xccl DEVICE=xpu \
  ./run_mpi.sh ddp_xpu_n4_inputs ../example/train_ddp.py
```

`run_mpi.sh` starts one LTTng userspace session, then launches `mpiexec -n NRANKS`
with `LD_PRELOAD` and the `TRACER_*` toggles exported so they reach every rank. All
ranks register with the same session, so their events land in one trace, separated
by `vpid`.

## Where the traces are written

Each experiment lands in `$LTTNG_HOME/lttng-traces/rec-<label>-<timestamp>/`. Read
one with:

```bash
source env.sh lttng
tracer/view.sh "$(ls -dt $LTTNG_HOME/lttng-traces/rec-ddp_xpu_n4_global-* | head -1)"
```

The `traces/*.txt` files here are those babeltrace2 dumps, saved so the results are
readable without re-running.

## The results

### N ranks still produce N coherent `vpid`s — one tile each
Four XPU DDP ranks recorded into one shared session yield four processes, each with
an identical event count, each bound to its own GPU tile:

```
ddp_xpu_n4_global:  4624 events (2312 entry / 2312 exit), balanced=True
    rank 0 -> xpu:0     rank 1 -> xpu:1
    rank 2 -> xpu:2     rank 3 -> xpu:3
baseline ddp_xpu_n1:  1096 events, 1 vpid
```

Per-rank device binding (`torch.xpu.set_device(local_rank)`) is a workload detail;
the tracer sees only ordinary threads under each `vpid`. The process-coherence
result from Step 5 holds unchanged on device.

### The CPU tie breaks: backward runs on a dedicated autograd thread *per rank*
On CPU (Step 5) each rank was two threads and backward ran **inline** on the main
thread. On XPU each rank is again two threads — but now the second one is the
**autograd engine**, and it carries the entire backward pass:

```
ddp_xpu_n4_global: 8 vtids across 4 vpids
  per rank:  main     vtid  maxdepth=3  bwd=0    <- forward + dispatch, no backward
             autograd vtid  maxdepth=5  bwd=66   <- the whole backward graph
baseline n1:  main 116579 (bwd=0) + autograd 116607 (bwd=66)
```

The `*Backward0` / `autograd::engine::evaluate_function:*` events land on the
autograd vtid; the main thread has **zero** backward events. This is Step 4's
single-process device finding, now per-rank: the device autograd engine dispatches
the backward graph on its own thread, so the main thread's max nesting depth (3) is
shallower than CPU's (6) — backward's depth has moved to the engine thread (5).

### On this workload thread-local still captures 100% — and why that is *not* Step 4
This is the subtle result. Step 4 found a thread-local callback on XPU saw only
8.3% of a Hogwild trace. Here it sees **everything**:

```
global (ddp_xpu_n4_global): 4624 events, 8 vtids
local  (ddp_xpu_n4_local):  4624 events, 8 vtids
thread-local captured 4624/4624 = 100.0%   (missed 0.0%)
  local trace still carries all 4 autograd threads, bwd=66 each (264 autograd events, both modes)
```

The autograd thread is captured **even in thread-local mode** because torch
propagates its `ThreadLocalState` — which includes the RecordFunction thread-local
callback — to the autograd engine thread it spawns from the instrumented main
thread. Step 4's 91.7% miss came from **Python-spawned** Hogwild worker threads
(`N_THREADS=4`) that never inherited that state. DDP has no such user threads: each
rank is main + engine, and the engine inherits the callback. So the reach gap is a
property of *who spawns the thread* (torch vs. user Python), not of device alone —
`global` is still the safe default (it needs no inheritance), but on plain DDP it
is not the difference between 100% and 8%.

### The DDP collectives reach the hook, and device shows in the args
The `xccl` gradient synchronization surfaces as the same c10d dispatched ops seen
on CPU, and with `TRACER_INPUTS=1` every op renders the tile its tensors live on:

```
   24  c10d::allreduce_                 <- gradient averaging (xccl)
   32  c10d::broadcast_                 <- initial param / buffer sync
    8  c10d::allgather_
devices in args (ddp_xpu_n4_inputs): xpu:0=525  xpu:1=525  xpu:2=525  xpu:3=525  cpu=220
```

Each rank's ops carry its own tile (`xpu:0…3`, 525 each — perfectly symmetric); the
`cpu` args are the host-side scalars and the small control tensors. Collective
communication and device placement are both visible with no extra instrumentation —
the multi-process, on-device analogue of "device shows up in the args for free".

## What this demonstrates

- **Process coherence holds on device.** Four XPU ranks → four balanced `vpid`s
  (one GPU tile each) in one session, same `.so`, no tracer change.
- **The CPU tie breaks per-rank.** Backward moves off the main thread onto a
  dedicated autograd engine thread in **every** rank (bwd=66 on the engine vtid,
  0 on main) — Step 4's device effect, now multiplied across ranks.
- **Thread-local reach is about the spawner, not the device.** On plain DDP
  thread-local still captured 100%, because torch propagates the callback to the
  autograd thread it spawns; the Step 4 miss was from user Python threads that
  don't inherit it. `global` remains the safe default.
- **Collectives + device are captured for free.** `c10d::allreduce_`/`broadcast_`/
  `allgather_` bracket like any op, and `INPUTS=1` shows each rank's ops on its own
  `xpu:r` tile — communication and placement visible with no new instrumentation.
- **The design holds across the full matrix.** One `.so` plus the LTTng `vpid`/
  `vtid` contexts scales from one process to many and from CPU to device, with
  balance, depth, and attribution intact.
