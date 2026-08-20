# Step 5 — How the granularities behave across MPI processes (DDP)

Steps 2–4 traced RecordFunction within **one process** — single-thread, then CPU
threads, then a GPU device runtime. This step adds the axis real training actually
uses: **multiple processes** running `DistributedDataParallel` (DDP) under
`mpiexec`, one model replica per rank, gradients averaged by collective
communication every backward pass. The question is whether RecordFunction + LTTng
stays coherent when the same program runs as N OS processes — and what the hook
sees of the collectives that tie them together. This CPU pass (gloo backend, 4
ranks on one Aurora compute node) answers the first: **N ranks produce N
independent, balanced `vpid` traces in one session, with no tracer change.** The
GPU pass (oneCCL) is a separate follow-up.

The tracer is **UNCHANGED from Step 4 — the same `.so`.** MPI is a *launch* axis,
not a tracer change: the tracer has zero MPI awareness, and LTTng's `vpid` context
(already recorded since Step 3) is what separates the ranks.

| Toggle | Values | Effect |
|---|---|---|
| `NRANKS` | `1` \| `4` (default) | number of MPI ranks `mpiexec` spawns (processes = `vpid`s) |
| `BACKEND` | `gloo` (default) \| `ccl` \| `mpi` | `torch.distributed` collective backend; `gloo` = CPU |
| `TRACER_THREAD` | `global` (default) \| `local` | register callback on every thread vs the load thread only |
| `TRACER_INPUTS` | `0` (default) \| `1` | render `fn.inputs()` — exposes the collectives' tensor args |

`NRANKS` is the new axis; the rest carry over from Steps 2–4. A **compute node is
required** — Cray PALS `mpiexec` will not launch on a login node (no `PBS_JOBID` →
exit 127); reach one with `qsub -A Performance -q debug -l walltime=00:60:00 -l
filesystems=home -l select=1 -I`.

## Contents

```
05_mpi_multiprocess/
├── README.md                    # this file
├── env.sh                       # module recipe (same across stages)
├── example/
│   └── train_ddp.py             # DDP: N ranks, one replica each, allreduce every backward
├── tracer/
│   ├── tracer.cpp               # UNCHANGED from Step 4 — same .so
│   ├── pytorch_tracepoints.tp   # LTTng-UST schema: name + scope + depth + args
│   ├── build.sh                 # lttng-gen-tp + compile -> libtorch_tracer.so
│   ├── run.sh                   # single-process recorder (Steps 1–4)
│   ├── run_mpi.sh               # NEW: mpiexec -n N into ONE shared LTTng session
│   └── view.sh                  # babeltrace2 <trace-dir>
└── traces/                      # babeltrace2 dumps
    ├── ddp_cpu_n1.txt           # baseline, 1 rank, global          (1196 events, 2 vtid, 1 vpid)
    ├── ddp_cpu_n4_global.txt    # 4 ranks, global callback          (5000 events, 8 vtid, 4 vpid)
    ├── ddp_cpu_n4_local.txt     # 4 ranks, thread-local callback    (5000 events, 8 vtid, 4 vpid)
    └── ddp_cpu_n4_inputs.txt    # 4 ranks, global, INPUTS=1         (5000 events, 4 vpid, collective args)
```

## How to reproduce

```bash
# On an Aurora COMPUTE node (login node has no PALS launcher — mpiexec exits 127).
source env.sh lttng                       # oneapi -> lttng -> babeltrace2 -> frameworks (LAST)
export LTTNG_HOME="/tmp/lttng_${USER}_$$"  # node-local; avoids shared-home sessiond lock
cd tracer && ./build.sh                    # -> tracer/libtorch_tracer.so (same source as Step 4)

# baseline: single rank (must match a single-process run)
NRANKS=1 TRACER_SCOPES=function+backward TRACER_THREAD=global BACKEND=gloo DEVICE=cpu \
  ./run_mpi.sh ddp_cpu_n1 ../example/train_ddp.py

# focus: 4 ranks into ONE shared session -> expect 4 vpids
NRANKS=4 TRACER_SCOPES=function+backward TRACER_THREAD=global BACKEND=gloo DEVICE=cpu \
  ./run_mpi.sh ddp_cpu_n4_global ../example/train_ddp.py

# thread-local variant, and collective tensor args
NRANKS=4 TRACER_SCOPES=function+backward TRACER_THREAD=local  BACKEND=gloo DEVICE=cpu \
  ./run_mpi.sh ddp_cpu_n4_local ../example/train_ddp.py
NRANKS=4 TRACER_SCOPES=function+backward TRACER_INPUTS=1      BACKEND=gloo DEVICE=cpu \
  ./run_mpi.sh ddp_cpu_n4_inputs ../example/train_ddp.py
```

`run_mpi.sh` starts one LTTng userspace session, then launches `mpiexec -n NRANKS`
with `LD_PRELOAD` and the `TRACER_*` toggles exported so PALS forwards them to every
rank. All ranks register with the same session, so their events land in one trace,
separated by `vpid`.

## Where the traces are written

Each experiment lands in `$LTTNG_HOME/lttng-traces/rec-<label>-<timestamp>/`. Read
one with:

```bash
source env.sh lttng
tracer/view.sh "$(ls -dt $LTTNG_HOME/lttng-traces/rec-ddp_cpu_n4_global-* | head -1)"
```

The `traces/*.txt` files here are those babeltrace2 dumps, saved so the results are
readable without re-running.

## The results

### N ranks produce N coherent `vpid`s — for free, no tracer change
Four gloo DDP ranks recorded into one shared session yield exactly four processes,
each with an **identical event count** — symmetric SPMD, as expected:

```
ddp_cpu_n4_global:  5000 events (2500 entry / 2500 exit), balanced=True
    1250 events  vpid 34854   <- rank 0
    1250 events  vpid 34855   <- rank 1
    1250 events  vpid 34856   <- rank 2
    1250 events  vpid 34857   <- rank 3
baseline ddp_cpu_n1:  1196 events, 1 vpid
```

The tracer has no notion of rank or MPI; LTTng's `vpid` context (recorded since
Step 3) is the whole mechanism. `mpiexec` forwarded `LD_PRELOAD` to every rank, so
each process was injected and traced independently. The process axis is the clean
analogue of Step 3's per-thread result.

### Every rank is internally balanced with its own depth stack — no cross-rank bleed
Within each `vpid`, entry/exit balance and the nesting-depth invariant hold exactly
as in a single-process run:

```
vtid 34854 (rank0 main):  entry=611 exit=611  maxdepth=6   BALANCED
vtid 34855 (rank1 main):  entry=611 exit=611  maxdepth=6   BALANCED
vtid 34856 (rank2 main):  entry=611 exit=611  maxdepth=6   BALANCED
vtid 34857 (rank3 main):  entry=611 exit=611  maxdepth=6   BALANCED
vtid 34874/75/78/79 (per-rank worker):  entry=14 exit=14  maxdepth=1  BALANCED
-> 8 distinct vtids across 4 vpids, ZERO overlap
```

Each rank runs two threads — a main thread (`vtid == pid`) carrying the full
forward+backward step, and one short-lived worker (the DDP reducer / gloo helper).
No `vtid` appears under two `vpid`s: the `thread_local` depth counter stays
process-private, so the multi-process trace is fully attributable. On CPU the
backward pass runs **inline on each rank's main thread** (the 66 backward events
sit on `vtid 34854…`, not a separate thread) — consistent with Steps 2–3, and the
reason global-vs-thread-local does **not** diverge here (next finding).

### On CPU, thread-local loses nothing — the tie of Step 2 reappears per-rank
Unlike Steps 3–4, where a thread-local callback missed 91% of the work, here the
two registration modes are byte-for-byte equal:

```
global (ddp_cpu_n4_global): 5000 events, 8 vtids
local  (ddp_cpu_n4_local):  5000 events, 8 vtids
thread-local captured 5000/5000 = 100.0%   (missed 0.0%)
```

Because CPU backward is inline and DDP's reducer work is dispatched from the same
main thread, nothing important leaves the load thread's reach. This is expected on
CPU and is precisely the tie the GPU pass is designed to break (a device autograd
engine thread would again make `global` mandatory).

### The DDP collectives ARE dispatched ops the hook brackets
With four ranks, `train_ddp.py`'s gradient synchronization surfaces as real
RecordFunction events — the c10d collective layer dispatches through the same seam
as any ATen op:

```
   24  c10d::allreduce_                              <- gradient averaging
   32  c10d::broadcast_                              <- initial param / buffer sync
    8  c10d::allgather_
   96  torch.distributed.ddp.reducer::copy_bucket_to_grad
   96  torch::distributed::reducer::mul_out
   24  aten::broadcast_tensors
```

With `TRACER_INPUTS=1` the collectives render their argument shapes
(`list, Object, scalar, …` — the bucketed gradient tensors + process-group
handle). This is the multi-process analogue of "device shows up in the args for
free": **collective communication is visible to RecordFunction with no extra
instrumentation.** What that entry/exit interval *means* on the wire (launch vs
actual transfer) is the focus of the next finding set, deferred to the GPU pass
where communication is not loopback.

## What this demonstrates

- **Multi-process tracing is per-`vpid` coherent for free.** Four ranks → four
  balanced `vpid`s (1250 events each) in one session, with no tracer change — the
  process-level analogue of Step 3's per-thread result.
- **No cross-rank contamination.** 8 vtids across 4 vpids with zero overlap; the
  `thread_local` depth stack stays process-private, so every rank's trace is
  independently attributable.
- **On CPU, global and thread-local tie again.** Backward is inline and reducer
  work stays on the main thread, so thread-local captured 100% — the Step 2 tie,
  reappearing per-rank. The GPU pass is what breaks it.
- **DDP collectives reach the hook.** `c10d::allreduce_` / `broadcast_` /
  `allgather_` and the reducer ops are dispatched ATen ops, bracketed like any
  other and carrying tensor args under `INPUTS=1` — communication structure is
  captured with no new instrumentation.
- **The design holds across processes.** The same `.so`, the same LTTng recipe plus
  a `vpid` context, scales from one process to many with balance and depth intact.
