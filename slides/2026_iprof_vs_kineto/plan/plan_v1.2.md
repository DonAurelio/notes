# Plan v1.2 — Full Re-Run with Corrected Event Counting

## Context

Phase 1 validation (exp_v1.1) established:

| Tool | Raw recording | n_events definition |
|---|---|---|
| iprof | 2 LTTng events per call (entry + exit) + metadata events per run | `babeltrace2 <raw_trace> | wc -l` immediately after collection |
| Kineto | 1 FunctionEvent per call (entry fills start_ns, exit fills end_ns on same object) | `sum(e.count for e in prof.key_averages())` — already correct |

Formula correction (2 × intervals + metadata(S)) was rejected: metadata event count
varies with (S, N) due to JIT compilation and kernel specialization. Only a direct
babeltrace2 count per rep is reliable.

## Changes from v1.0

| | v1.0 | v1.2 |
|---|---|---|
| iprof n_events | tally intervals (awk on replay output) | `babeltrace2 raw_trace | wc -l` inline after collection |
| Kineto n_events | `sum(e.count for e in avgs)` | same — already correct |
| Replay phase | separate Phase 2, awk-patches raw.csv | inline per rep, complete row written at once |
| S=4096, N=1000 | excluded | included |
| Summary output | median + IQR, one table | mean + std, two tables (pandas) |

## Parameters

S ∈ {256, 1024, 4096} × N ∈ {10, 100, 1000} = 9 cells × 15 reps = 135 runs per scenario,
405 total. Warmup = 5.

## Script flow (run_xpu.sh)

Single loop per (S, N, rep):
1. **BL**: run bench.py → append row
2. **IP**: collect raw trace → `babeltrace2 | wc -l` → `iprof --replay` for analysis_s → append complete row
3. **KN**: run bench_kineto.py → append row (n_events + analysis_s from script output)

## Output tables

**results/summary_thot.csv** — T_hot mean ± std, one row per (S, N), scenarios as columns:
```
S, N, BL_mean, BL_std, IP_mean, IP_std, KN_mean, KN_std
```

**results/summary_timing.csv** — all timing metrics mean ± std, one row per (scenario, S, N):
```
scenario, S, N, T_hot_s_mean, T_hot_s_std, T_process_s_mean, T_process_s_std, analysis_s_mean, analysis_s_std
```

## Planned visualizations (viz/)

1. **overhead_ratio.png** — (T_hot_traced − T_hot_BL) / T_hot_BL [%] vs N, by tool, faceted by S
2. **analysis_per_event.png** — analysis_s / n_events [ns/event] vs N, by tool, faceted by S
3. **n_events_comparison.png** — n_events vs N for iprof and Kineto, faceted by S
