# Tracing Overhead Evaluation Plan: iprof vs Kineto
## Version 1.5

---

## Goals

Compare tracing overhead between **iprof** (THAPI, default mode) and **Kineto** on XPU workloads, tracing both CPU dispatch activity and GPU kernel execution. The comparison must separately characterize the fixed startup cost and the per-iteration hot-region cost.

---

## Requirements

- All experiments must run on a compute node with XPU devices.
- Tensors must be allocated on XPU and both tools must trace the same set of operations: allocation, warm-up, and hot region.
- `torch.xpu.synchronize()` must be called before stopping any timer so async GPU work is included in the measured time.
- iprof collection (`--no-analysis`) and analysis (`--replay`) must run as separate steps so their costs are measured independently.
- iprof must run in default mode.
- Run a short pilot (3 reps, S=1024, N=100, all scenarios) at the start of the allocation to validate per-run wall times.

---

## Metrics

### A. Hot-region execution overhead

```
overhead_ratio    = (T_hot_traced − T_hot_BL) / T_hot_BL × 100    [%]
overhead_per_iter = (T_hot_traced − T_hot_BL) / N_iters             [ms/iter]
```

`T_hot` = inner timer wall time of the N hot iterations, printed by the script.

### B. Fixed startup overhead

```
startup_overhead = (T_process_traced − T_process_BL) − (T_hot_traced − T_hot_BL)   [s]
```

`T_process` = external wall time of the full process. Isolates the LTTng session cost (iprof) or profiler initialization cost (Kineto).

### C. Analysis/serialization time

| Tool | Measured quantity | How |
|---|---|---|
| iprof | Offline tally (Babeltrace2) | `time iprof --replay <trace_dir>` after collection |
| Kineto | `prof.key_averages()` | Inner timer in script, printed as `analysis_s=` |

Also reported: `analysis_per_event = T_analysis / N_events` [ns/event].

### D. Supporting (raw data only)

`N_events` from full execution (allocation + warm-up + hot), reported per scenario for context.

---

## Workload

`torch.matmul(x, x)` on an XPU tensor of shape (S, S), with W=5 warm-up iterations followed by N hot iterations.

```python
# bench.py — used for BL and iprof
import torch, time, argparse

p = argparse.ArgumentParser()
p.add_argument("--size",  type=int, required=True)
p.add_argument("--iters", type=int, required=True)
args = p.parse_args()

x = torch.randn(args.size, args.size, device="xpu")  # traced by iprof

for _ in range(5):                      # warm-up
    torch.matmul(x, x)
torch.xpu.synchronize()

t0 = time.perf_counter()
for _ in range(args.iters):             # hot region
    torch.matmul(x, x)
torch.xpu.synchronize()
t1 = time.perf_counter()

print(f"wall_s={t1-t0:.6f}")
```

```python
# bench_kineto.py — allocation inside context manager to match iprof scope
import torch, time, argparse
from torch.profiler import profile, ProfilerActivity

p = argparse.ArgumentParser()
p.add_argument("--size",  type=int, required=True)
p.add_argument("--iters", type=int, required=True)
args = p.parse_args()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
    x = torch.randn(args.size, args.size, device="xpu")  # inside: matches iprof scope

    for _ in range(5):                  # warm-up
        torch.matmul(x, x)
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.iters):         # hot region
        torch.matmul(x, x)
    torch.xpu.synchronize()
    t1 = time.perf_counter()

print(f"wall_s={t1-t0:.6f}")

t_a0 = time.perf_counter()
avgs = prof.key_averages()
t_a1 = time.perf_counter()
print(f"analysis_s={t_a1-t_a0:.6f}")
print(f"n_events={sum(e.count for e in avgs)}")
```

---

## Parameters

| Parameter | Values | Why |
|---|---|---|
| Matrix size (S) | 256, 1024, 4096 | Controls compute time per iteration; separates fixed from per-iter cost |
| Iteration count (N) | 10, 100, 1000 | Exposes amortization of fixed startup cost across a 100× range |

**S=4096, N=1000 excluded**: estimated ~41 min for one cell; amortization already covered by (S=4096, N=100) and (S=1024, N=1000). Gives **8 parameter combinations**.

iprof is run in **default mode** throughout.

---

## Scenarios

| ID | Tool | iprof backends | Kineto activities | Command |
|---|---|---|---|---|
| BL-XPU | None | — | — | `python3 bench.py --size S --iters N` |
| IP-XPU | iprof | `pytorch,ze` | — | `iprof --no-analysis --backends pytorch,ze -- python3 bench.py ...` |
| KN-XPU | Kineto | — | `CPU + XPU` | `python3 bench_kineto.py --size S --iters N` |

---

## Run Matrix and Time Estimate

3 scenarios × 8 parameter combinations × 15 repetitions = **360 runs**, plus 8 × 15 iprof replay runs for analysis time. Based on CPU pilot data (iprof full-process cost ~10–15 s/run), the full experiment is estimated at **45–60 min and fits in one 60-min compute allocation**.

---

## Measurement Procedure

1. Reserve compute node: `qsub -A Performance -q debug -l walltime=00:60:00 -l filesystems=home -I -l select=1`.
2. Run the XPU pilot (S=1024, N=100, 3 reps per scenario); verify wall times before proceeding.
3. Run all BL-XPU, IP-XPU (collection), and KN-XPU in the same allocation.
4. Per run: capture `T_hot` from stdout; capture `T_process` via external `time` wrapper.
5. After all IP-XPU collection: replay each trace with `time iprof --replay <dir>` → record `T_analysis`.
6. Compute per-cell statistics (median, IQR) across 15 reps; derive summary metrics from medians.

Environment per SSH call:
```bash
module use /home/avivasmeza/thapi_env/modules && module load thapi-env/blob && module load frameworks && <command>
```

---

## Data Organization

```
iprof_vs_kineto/
  workloads/
    bench.py               # BL and iprof workload
    bench_kineto.py        # Kineto workload
  scripts/
    run_xpu.sh             # drives all cells via SSH, appends to raw log
    collect.py             # parses raw logs → raw.csv
    summarize.py           # raw.csv → summary.csv
  traces/
    IP-XPU/<S>_<N>/<rep>/  # iprof CTF traces
  results/
    raw.csv
    summary.csv
  analysis/
    plot_overhead.py
  plan/
    plan_v1.0.md
    plan_v1.3.md
    plan_v1.4.md
    plan_v1.5.md           # this document
```

### raw.csv columns

| Column | Type | Description |
|---|---|---|
| `scenario` | str | `BL-XPU`, `IP-XPU`, `KN-XPU` |
| `S` | int | Matrix side length |
| `N` | int | Hot-region iteration count |
| `rep` | int | Repetition index (1–15) |
| `T_hot_s` | float | Hot-region wall time from inner timer [s] |
| `T_process_s` | float | Full process wall time from external `time` [s] |
| `n_events` | int | Total events (allocation + warm-up + hot); null for BL |
| `analysis_s` | float | Tally/serialization time [s]; null for BL |

### summary.csv — derived metrics (median + IQR per cell)

| Column | Definition |
|---|---|
| `overhead_ratio` | `(T_hot_traced − T_hot_BL) / T_hot_BL × 100` [%] |
| `overhead_per_iter` | `(T_hot_traced − T_hot_BL) / N` [ms/iter] |
| `startup_overhead_s` | `(T_process_traced − T_process_BL) − (T_hot_traced − T_hot_BL)` [s] |
| `analysis_per_event` | `analysis_s / n_events` [ns/event] |

---

## Hypotheses

| Hypothesis | Direction | Reasoning |
|---|---|---|
| iprof `startup_overhead_s` | ≥7.5 s constant | ~7.5 s confirmed on CPU pilot; ZE backend may add slightly more |
| iprof `overhead_per_iter` increases with S | Expected | CPU pilot: 8 μs/iter at S=256 → 6.7 ms/iter at S=4096 |
| iprof `overhead_ratio` decreases as N increases (same S) | Decreasing | Fixed startup cost amortizes over longer hot regions |
| Kineto `startup_overhead_s` | Near 0 | In-process; no external session startup |
| Kineto `overhead_per_iter` vs iprof | Lower for large S | No LTTng interception cost; XPU device-side timing adds some overhead |
| iprof vs Kineto `analysis_per_event` | iprof higher | Babeltrace2 offline CTF parsing is heavier than in-memory Kineto aggregation |
