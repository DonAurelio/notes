# Investigation 1.1 — Minimal reproducer for Kineto ZE event loss in DDP

Extends inv_1.0. That investigation identified the root cause using the full
TinyLlama workload. This one narrows it to a self-contained ~60-line script
with no project dependencies, and pins down exactly which call is the trigger
and whether distributed compute across nodes is required.

---

## Summary

| Scope | n_events | ze_events |
|---|---|---|
| Wide (profile before DDP init) | 239 | 30 |
| Narrow (profile after DDP init) | 1,497 | 1,416 |

Opening `profile()` before `DistributedDataParallel.__init__()` under an
mpiexec-launched xccl process group causes a **98% drop in ZE events**
(30 → 1,416). Reproduced on a **single node** with `mpiexec -n 2 -ppn 2`.

---

## What is required to trigger the bug

| Condition | Required? | Evidence |
|---|---|---|
| Two ranks via `mpiexec` | **Yes** | torchrun single-node: no bug |
| Cross-node (`-ppn 1`) | **No** | single-node `-ppn 2` also triggers it |
| `DistributedDataParallel.__init__` | **Yes** (trigger) | cp=1 (after init_pg only): ze=30; cp=2 (after DDP init): ze=1,480 |
| Collective ops (`all_reduce`) | **No** | the trigger is DDP init, not the first collective |
| `MASTER_ADDR` set to rank-0 host | **Yes** (infra) | loopback `127.0.0.1` hangs init_process_group across ranks |

The minimal condition is: **mpiexec + xccl + profile() open before DDP.__init__()**.
Single node is sufficient. Cross-node is not required.

---

## Bisect results

Five checkpoints, same step loop (matmul + all_reduce), single node `-ppn 2`.
First checkpoint with `ze_events > 0` is the call that unblocks ZE for Kineto.

| cp | profile opens after | ze_events (1-node) | ze_events (2-node) |
|---|---|---|---|
| 0 | — (before everything) | 30 | 30 |
| 1 | `dist.init_process_group` | 30 | 30 |
| **2** | **`DDP.__init__`** | **1,480** | **1,516** |
| 3 | first `all_reduce` | 1,342 | 1,467 |
| 4 | first `torch.matmul` | 209 | 308 |

**`DDP.__init__` is the trigger.** `dist.init_process_group` alone (cp=1) is
not enough — ZE is still not reachable by the Kineto listener after CCL's
bootstrap. It is the parameter broadcast inside `DDP.__init__`
(`_broadcast_coalesced`) that drives the first real XPU kernel dispatch through
CCL, completing ZE initialization in a way the listener can hook into.

---

## Why `torchrun` is not affected

With `torchrun --nproc-per-node=2`, both ranks share a PMI-free process group
setup. PyTorch's own XPU runtime initializes ZE on the first
`tensor.to(device)` call, and the Kineto hook attaches at that point via
PyTorch's internal ZE context — not through CCL's separate initialization
path. The CCL/mpiexec path is what creates the race condition.

---

## How the error is produced — step by step

```
mpiexec -n 2 -ppn 2 -hosts HOST
  └─ rank 0                              └─ rank 1
     profile().__enter__()                  profile().__enter__()
       → profiler_start fires                → profiler_start fires
       → XPU listener tries to attach         → XPU listener tries to attach
         ZE context not yet initialized         ZE context not yet initialized
         registration FAILS silently           registration FAILS silently

     dist.init_process_group(xccl)         dist.init_process_group(xccl)
       → CCL bootstrap (ZE partial init)     → CCL bootstrap (ZE partial init)
       [listener still not attached]         [listener still not attached]

     DDP.__init__()                         DDP.__init__()
       → _broadcast_coalesced                → _broadcast_coalesced
         first real XPU kernel dispatch        first real XPU kernel dispatch
         ZE fully initialized                  ZE fully initialized
       [too late — listener never attached]  [too late — listener never attached]

     ── step loop ──
     torch.matmul, all_reduce → ZE kernel dispatches NOT captured
     profile().__exit__() → ze_events = 30   (noise / CCL bookkeeping only)
```

The USDT probe in stderr pinpoints the timing:
```
USDT:... SyncActivityProfilerHandler.cpp:52] profiler_start   ← listener fails here
... init_process_group + DDP.__init__ ...                     ← ZE comes up here
USDT:... SyncActivityProfilerHandler.cpp:59] profiler_stop    ← session ends
```

---

## Reproducer

**File:** `workloads/repro_kineto_ddp.py`

~60 lines. No external dependencies beyond `torch`. No `mpi4py`,
no `llama3_demo_common`, no `ccl_local_wrap.sh`.

Reads rank/world_size directly from PALS env vars set by mpiexec
(`PALS_RANKID`, `PALS_WORLD_SIZE`, `PALS_LOCAL_RANKID`), with fallbacks
for PMI and Open MPI variables.

```python
# wide: profile before DDP init — broken
if args.scope == "wide":
    prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU])
    prof.__enter__()

dist.init_process_group(backend="xccl", init_method="env://", ...)
model = nn.Linear(64, 64).to(device)
model = DistributedDataParallel(model, device_ids=[local_rank], ...)
torch.xpu.synchronize()

# narrow: profile after DDP init — correct
if args.scope == "narrow":
    prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU])
    prof.__enter__()

for _ in range(steps):
    model(torch.randn(64, 64, device=device))
    dist.all_reduce(torch.ones(1, device=device))
    torch.xpu.synchronize()
```

---

## Run scripts

| Script | Nodes | Purpose |
|---|---|---|
| `run_repro_final.sh` | 1 (`select=1`, `-ppn 2`) | Final minimal reproducer |
| `run_bisect_singlenode.sh` | 1 (`select=1`, `-ppn 2`) | 5-checkpoint bisect, single node |
| `run_bisect.sh` | 2 (`select=2`, `-ppn 1`) | 5-checkpoint bisect, cross-node |

All require `MASTER_ADDR=$HOST0` exported before mpiexec. No `ccl_local_wrap.sh` needed.

---

## Results

### Final reproducer — job 8978739, 2026-10-28, single node `x4102c6s3b0n0`

```
scope=wide    n_events=239    ze_events=30
scope=narrow  n_events=1,497  ze_events=1,416
```

### Bisect — job 8978215 (2-node) and 8978275 (1-node)

| cp | opens after | ze (1-node) | ze (2-node) |
|---|---|---|---|
| 0 | before init_process_group | 30 | 30 |
| 1 | after init_process_group | 30 | 30 |
| 2 | after DDP.__init__ | 1,480 | 1,516 |
| 3 | after first all_reduce | 1,342 | 1,467 |
| 4 | after first matmul | 209 | 308 |

---

## Conclusion

The Kineto XPU listener fails when `profile()` opens before
`DistributedDataParallel.__init__()` under any mpiexec-launched xccl process
group — single node or multi-node. The trigger is CCL's ZE initialization
path driven by `DDP.__init__`'s parameter broadcast, not the collective ops
in the training loop, and not `dist.init_process_group` alone.

The fix for the main experiment: open `profile()` after
`initialize_process_group()` + `DDP.__init__()` + `synchronize()`.
