# Investigation 1.0 — Kineto n_events decreases with wider profile() scope

## Observation

During smoke tests of the DDP iprof vs Kineto experiment, two successive runs with
different `profile()` scopes produced contradictory event counts:

### Smoke run 1 — narrow scope (profile wraps step loop only)

| Scenario | T_hot (s) | T_process (s) | n_events | analysis (s) |
|---|---|---|---|---|
| BL | 0.699 | 6.56 | — | — |
| iprof | 0.768 | 22.11 | 114,042 | 2.53 |
| Kineto | 0.714 | 6.01 | **20,596** | 0.838 |

### Smoke run 2 — wide scope (profile opens after select_device, before DDP init)

| Scenario | T_hot (s) | T_process (s) | n_events | analysis (s) |
|---|---|---|---|---|
| BL | 0.646 | 4.95 | — | — |
| iprof | 0.763 | 19.37 | 114,038 | 1.22 |
| Kineto | 0.675 | 5.93 | **7,625** | 0.308 |

---

## Question

**Why does a wider `profile()` scope produce fewer `sum(e.count for e in prof.key_averages())` events?**

The wider scope captures everything the narrow scope does, plus DDP init, model
construction, and optimizer setup. Naively, more code under the profiler should mean
more events — yet n_events drops from 20,596 to 7,625.

---

## Background: what n_events measures

From PyTorch source (`torch/autograd/profiler_util.py`):

- Each raw `FunctionEvent` has `count = 1` (one recorded operator invocation).
- `key_averages()` groups events by `(op_name, device_type, node_id, ...)` into `FunctionEventAvg` objects.
- `FunctionEventAvg.count` accumulates: `self.count += other.count` — so it equals the number of raw invocations of that op.
- `sum(e.count for e in prof.key_averages())` == total raw operator invocations == `len(prof.events())`.

A wider scope should add more invocations, not remove them.

---

## Results

### Event counts

| Scope | n_events |
|---|---|
| Narrow (profile wraps step loop only) | 20,597 |
| Wide (profile opens before DDP init) | 7,625 |
| **Difference** | **−12,972** |

### Where the 12,972 missing events went

**Level Zero (ZE) GPU driver calls — most affected.** The wide scope loses nearly all of them because the XPU activity listener never attaches. For example, `zeKernelSetArgumentValue` drops from 5,512 to 0 — one call per kernel argument, per launch, across all 5 steps.

| op | narrow | wide |
|---|---|---|
| `zeKernelSetArgumentValue` | 5,512 | 0 |
| `zeCommandListAppendLaunchKernel` | 1,171 | 0 |
| `zeKernelSetGroupSize` | 1,171 | 0 |
| `zeDeviceGetRootDevice` | 864 | 0 |
| `zeEventHostReset` | 1,057 | 11 |
| `zeMemGetAllocProperties` | 924 | 12 |

**ATen ops — essentially equal.** The Python-level operator counts are the same in both scopes because they are recorded by the CPU-side profiler hooks, which attach regardless of when the profiler opens. Minor differences (e.g. `aten::copy_` 300 narrow vs 330 wide) reflect one-time init copies captured only in the wider window.

| op | narrow | wide |
|---|---|---|
| `aten::mm` | 90 | 90 |
| `aten::linear` | 60 | 60 |
| `c10d::allreduce_` | 6 | 6 |
| `aten::copy_` | 300 | 330 |
| `aten::empty` | 346 | 365 |

The gap is dominated by **ZE-level GPU driver events absent from the wide scope**:

| count (narrow) | op |
|---|---|
| 5,512 | `zeKernelSetArgumentValue` |
| 1,171 | `zeCommandListAppendLaunchKernel` |
| 1,171 | `zeKernelSetGroupSize` |
| 864 | `zeDeviceGetRootDevice` |
| 162 | `zeKernelGetProperties` |
| … | + 90 more XPU kernel/driver ops |

The 125 ATen-level ops (forward/backward passes, optimizer, collectives) are **identical or nearly identical** in count between narrow and wide — confirming the training logic itself was captured correctly in both scopes.

### Root cause

**Opening the Kineto profiler before ZE device initialization prevents the XPU activity listener from attaching.**

The XPU Kineto backend registers its device-activity listener at `profiler_start`
(`SyncActivityProfilerHandler.cpp:52`). In the wide scope, `profiler_start` fires
**before `initialize_process_group`**, which is when the Level Zero command queue is
fully set up. The listener opens against a not-yet-initialized device context and
**silently fails to attach** — so all `ze*` and XPU kernel events are absent for the
entire session.

In the narrow scope, the profiler opens **after** the full ZE device stack is
initialized (DDP init + `synchronize(device)` complete first). The listener attaches
correctly and captures every GPU-side kernel dispatch event.

### Conclusion

The narrow scope is **correct**: it captures real GPU activity. The wide scope is
**silently incomplete**: it loses all device-side events, making Kineto appear cheaper
than it is.

For the DDP experiment, `profile()` must open **after** `initialize_process_group` and
`synchronize(device)` to ensure the XPU activity listener attaches. The current
`train_llama3_kineto.py` (wide scope, opening before DDP init) is wrong and must be
fixed to match the narrow scope.

---

## Method

Run both variants back-to-back on the same 2-node allocation with identical args.
Dump the full `key_averages()` breakdown (op name + count) for each.
`diff_breakdown.py` identifies:
- Ops present only in narrow (not in wide)
- Ops present only in wide (not in narrow)
- Ops in both with different counts

The diff will show exactly which ops account for the 20,596 − 7,625 = **12,971 event gap**.

---

