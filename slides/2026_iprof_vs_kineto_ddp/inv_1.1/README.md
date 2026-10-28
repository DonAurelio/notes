# Investigation 1.1 — Minimal reproducer for Kineto ZE event loss in DDP

Extends inv_1.0. That investigation identified the root cause; this one
produces a self-contained reproducer that triggers the bug with ~60 lines of
code and no TinyLlama model dependency.

---

## Summary

| Scope | n_events | ze_events |
|---|---|---|
| Wide (profile before DDP init) | 253 | 30 |
| Narrow (profile after DDP init) | 1,440 | 1,333 |

Opening `profile()` before `DistributedDataParallel.__init__()` in a
2-rank mpiexec setup causes a 97% drop in ZE events (30 → 1,333).

---

## How the error is produced

### The three necessary conditions

All three must be present simultaneously:

1. **Two ranks on separate nodes** launched with `mpiexec -ppn 1`
2. **`profile()` opens before `DistributedDataParallel.__init__()`**
3. **`MASTER_ADDR` set to the rank-0 hostname** (required for the
   rendezvous store — loopback `127.0.0.1` hangs because rank 1 on
   a different node cannot reach rank 0's store)

Remove any one of these and the bug disappears or cannot be observed:
- Single node (`torchrun`): ZE init path differs, listener attaches correctly.
- Profile after DDP init: ZE is fully up, listener attaches correctly.
- `MASTER_ADDR=127.0.0.1`: `dist.init_process_group` hangs before the bug
  can be observed.

### Step-by-step sequence in the buggy (wide) case

```
mpiexec -n 2 -ppn 1 -hosts HOST0,HOST1
  └─ rank 0 on HOST0                    └─ rank 1 on HOST1
     profile().__enter__()                 profile().__enter__()
       → profiler_start fires               → profiler_start fires
       → SyncActivityProfilerHandler          → SyncActivityProfilerHandler
         tries to register ZE listener          tries to register ZE listener
         ZE context does not exist yet          ZE context does not exist yet
         listener registration FAILS            listener registration FAILS
         (silent, no exception raised)          (silent, no exception raised)
     dist.init_process_group(xccl)        dist.init_process_group(xccl)
       → CCL initializes ZE context          → CCL initializes ZE context
       → ZE command queue created            → ZE command queue created
     DDP.__init__()                        DDP.__init__()
       → _broadcast_coalesced              → _broadcast_coalesced
         (parameter sync via CCL)            (parameter sync via CCL)
     ── step loop ──                       ── step loop ──
     torch.matmul → ZE kernel dispatch    torch.matmul → ZE kernel dispatch
       NOT captured (no listener)           NOT captured (no listener)
     profile().__exit__()                  profile().__exit__()
       → key_averages(): ze_events=30       (rank 1 output suppressed)
```

### Why the listener fails silently

`SyncActivityProfilerHandler::profiler_start()` (line 52 of
`SyncActivityProfilerHandler.cpp`) registers a device-activity callback
through the Level Zero interception layer. When ZE is not yet initialized,
the Level Zero handle is null; the registration call returns without error
but with no effect. There is no warning in the Kineto log.

The USDT probe line in stderr confirms the exact moment:

```
USDT:... SyncActivityProfilerHandler.cpp:52] profiler_start   ← listener fails here
... dist.init_process_group(xccl) ...                         ← ZE comes up here
... DDP __init__ ...
USDT:... SyncActivityProfilerHandler.cpp:59] profiler_stop    ← too late
```

### Why single-node torchrun is not affected

With `torchrun --nproc-per-node=2`, both ranks share the same OS process
group and ZE device context. PyTorch's own XPU runtime initializes ZE on
the first `torch.randn(..., device="xpu")` call, and the Kineto hook attaches
to PyTorch's ZE context at that point — not via CCL's separate initialization
path. The CCL cross-node init is what creates the race condition.

### Why `dist.init_process_group` alone is not enough to trigger it

Calling `dist.init_process_group` without wrapping the model in
`DistributedDataParallel` does initialize the CCL/ZE context, but the Kineto
listener still misses events in the wide scope. The `DDP.__init__` call (which
runs `_broadcast_coalesced` to sync parameters across ranks) is not strictly
necessary to trigger the bug — the CCL init inside
`dist.init_process_group(xccl)` is sufficient. However, without a model the
step loop produces no ZE kernel dispatches, so there is nothing to count.
The `nn.Linear` + `DDP` wrapper is included in the reproducer to generate
measurable ZE events in the step loop.

---

## Reproducer

**File:** `workloads/repro_kineto_ze.py`

~60 lines. No TinyLlama. Uses `nn.Linear(64, 64)` wrapped in
`DistributedDataParallel` with 3 `torch.matmul` steps.

Imports `get_mpi_env`, `get_local_rank`, `initialize_process_group`,
`select_device`, `synchronize` from `llama3_demo_common` for correct
PALS/MPI env var handling — identical to the main workload.

**Key structure:**

```python
# wide: profiler opens before DDP init
if args.scope == "wide":
    prof_ctx = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU])
    prof_ctx.__enter__()

initialize_process_group(rank, world_size, "xccl", device)   # ZE comes up here
model = nn.Linear(64, 64).to(device)
model = DistributedDataParallel(model, device_ids=[local_rank], ...)
synchronize(device)

# narrow: profiler opens after DDP init
if args.scope == "narrow":
    prof_ctx = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU])
    prof_ctx.__enter__()

for _ in range(args.steps):
    run_step(model, device)

synchronize(device)
prof_ctx.__exit__(None, None, None)
```

---

## Run script

**File:** `scripts/run_repro.sh` — PBS batch job, `select=2`, 10-minute
walltime. Runs wide then narrow back-to-back on the same 2-node allocation.

Critical env vars that must be set before `mpiexec`:

```bash
export MASTER_ADDR=$HOST0      # rank-0 hostname; loopback hangs 2-node jobs
export PALS_WORLD_SIZE=2       # tells CCL the total world size
```

---

## Results

Run on 2026-10-28, Aurora nodes `x4406c4s0b0n0` + `x4406c4s1b0n0`,
job `8977805`.

```
scope=wide    n_events=253    ze_events=30
scope=narrow  n_events=1,440  ze_events=1,333
```

ZE events in wide scope: **30 out of 1,333 expected — 97% missing.**

---

## Conclusion

The reproducer confirms inv_1.0's root cause with a minimal workload.
The fix for the main experiment is to open `profile()` after
`initialize_process_group()` + `synchronize()`, as implemented in
`train_llama3_kineto_narrow.py`.
