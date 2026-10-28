# Investigation 1.1 — Kineto ZE event loss: minimal reproducer

Opening `profile()` before `DistributedDataParallel.__init__()` under `mpiexec + xccl` causes a **98% drop in ZE events**. Single node is sufficient to reproduce — cross-node is not required.

```
scope=wide    n_events=239    ze_events=30
scope=narrow  n_events=1,497  ze_events=1,416
```

**inv_1.0** observed the drop in the full TinyLlama workload (n_events 20,596 → 7,625) when moving `profile()` from after the step loop to before DDP init. 

**Root cause:** under `mpiexec + xccl`, the ZE context is not fully accessible to the Kineto listener until `DDP.__init__` completes its parameter broadcast. `profiler_start` before that point silently registers against a ZE context that is not yet hookable, and no error is raised.

---

### Reproducer

**`workloads/repro_kineto_ddp.py`**

```
MASTER_ADDR=$HOST mpiexec -n 2 -ppn 2 python3 repro_kineto_ddp.py --scope wide
MASTER_ADDR=$HOST mpiexec -n 2 -ppn 2 python3 repro_kineto_ddp.py --scope narrow
```

Or submit `scripts/run_repro_final.sh` (PBS, `select=1`).

---

## Fix

Open `profile()` after `dist.init_process_group()` + `DDP.__init__()` + `synchronize()`. Applied in `train_llama3_kineto_narrow.py`.
