# Investigation 1.1 — Kineto ZE event loss: minimal reproducer

## Result

```
scope=wide    n_events=239    ze_events=30
scope=narrow  n_events=1,497  ze_events=1,416
```

Opening `profile()` before `DistributedDataParallel.__init__()` under
`mpiexec + xccl` causes a **98% drop in ZE events**. Single node is
sufficient to reproduce — cross-node is not required.

---

## How we found it

**inv_1.0** observed the drop in the full TinyLlama workload (n_events 20,596
→ 7,625) when moving `profile()` from after the step loop to before DDP init.
Root cause hypothesis: the Kineto XPU listener (`SyncActivityProfilerHandler`)
fails to attach when `profiler_start` fires before ZE is initialized.

**inv_1.1 step 1 — isolate the launcher.** We replaced TinyLlama with
`nn.Linear(64,64)` + `DistributedDataParallel` + 3 matmul steps, keeping
`mpiexec -n 2 -ppn 1` and `ccl_local_wrap.sh`. Bug reproduced (ze=30 vs
ze=1,333). Then dropped `ccl_local_wrap.sh` — still reproduced. Then dropped
`llama3_demo_common` — still reproduced. Final reproducer: pure torch +
stdlib, reads rank from PALS env vars.

**inv_1.1 step 2 — bisect the trigger.** Added 5 checkpoints: profile opens
before `init_process_group` (cp=0), after `init_process_group` (cp=1), after
`DDP.__init__` (cp=2), after first `all_reduce` (cp=3), after first `matmul`
(cp=4). Result:

| cp | opens after | ze_events |
|---|---|---|
| 0 | before init_process_group | 30 |
| 1 | after init_process_group | 30 |
| **2** | **after DDP.__init__** | **1,480** |
| 3 | after first all_reduce | 1,342 |
| 4 | after first matmul | 209 |

`dist.init_process_group` alone does not unblock ZE for Kineto. The trigger
is `DDP.__init__`, which calls `_broadcast_coalesced` — the first real XPU
kernel dispatch through CCL — completing ZE initialization in a way the
listener can hook into.

**inv_1.1 step 3 — isolate the launcher.** Replaced `-n 2 -ppn 1` (2 nodes)
with `-n 2 -ppn 2` (1 node, same host). Bug reproduced identically. Cross-node
is not required. `torchrun` on the same node does not trigger it because
torchrun does not use the CCL/mpiexec initialization path.

**Root cause:** under `mpiexec + xccl`, the ZE context is not fully
accessible to the Kineto listener until `DDP.__init__` completes its
parameter broadcast. `profiler_start` before that point silently registers
against a ZE context that is not yet hookable, and no error is raised.

---

## Reproducer

**`workloads/repro_kineto_ddp.py`** — ~60 lines, no external dependencies.

```
MASTER_ADDR=$HOST mpiexec -n 2 -ppn 2 python3 repro_kineto_ddp.py --scope wide
MASTER_ADDR=$HOST mpiexec -n 2 -ppn 2 python3 repro_kineto_ddp.py --scope narrow
```

Or submit `scripts/run_repro_final.sh` (PBS, `select=1`).

---

## Fix

Open `profile()` after `dist.init_process_group()` + `DDP.__init__()` +
`synchronize()`. Applied in `train_llama3_kineto_narrow.py`.
