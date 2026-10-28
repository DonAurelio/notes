# iprof vs Kineto — DDP Workload

**Date**: TBD  
**Platform**: Aurora (Intel XPU, 2 nodes × 1 rank/node)

### Setup

| Item | Detail |
|---|---|
| Workload (T_hot) | TinyLlama DDP training loop, 2 nodes, 1 rank/node |
| Model | `dim=64, layers=1, heads=4, vocab-size=128, seq-len=16, batch-size=1, bf16` |
| Steps | TBD (set after smoke test) |
| Repetitions | TBD |
| iprof (T_process, T_hot) | `--no-analysis --backends pytorch,ze`; `wall_s=` from rank-0 stdout |
| Kineto (T_process, T_hot) | `ProfilerActivity.CPU + XPU`; `wall_s=` from rank-0 stdout |
| iprof (T_analysis) | `iprof --replay <raw_trace>` |
| Kineto (T_analysis) | `prof.key_averages()` |
| iprof (n_events) | `babeltrace2 raw_trace \| wc -l` |
| Kineto (n_events) | `sum(e.count for e in prof.key_averages())` |
| Summary statistic | Mean across reps; error reported as sample standard deviation (σ) |

---

### T_hot overhead

> _To be filled after experiment._

<img src="viz/overhead_ratio.png" width="60%">

`T_hot` is the wall time of the training-step loop (all steps timed together). Overhead is reported as
`(T_hot_traced − T_hot_BL) / T_hot_BL × 100%`.

| Scenario | T_hot mean ± std (s) | Overhead |
|---|---|---|
| BL | — | — |
| iprof | — | — |
| Kineto | — | — |

---

### Process time

`T_process` is total wall time including Python startup, tracer initialization, model init, and the step loop.

| Scenario | T_process mean ± std (s) |
|---|---|
| BL | — |
| iprof | — |
| Kineto | — |

---

### Event counts

> _To be filled after experiment._

<img src="viz/n_events_comparison.png" width="60%">

`n_events` is deterministic across reps (std = 0 for all cells).

| Tool | n_events | events/step |
|---|---|---|
| iprof | — | — |
| Kineto | — | — |

---

### Analysis cost

> _To be filled after experiment._

<img src="viz/analysis_per_event.png" width="60%">

`analysis_s` is the time to post-process the trace (`iprof --replay` for iprof,
`prof.key_averages()` for Kineto).

| Tool | analysis_s mean ± std | ns/event | Scaling |
|---|---|---|---|
| iprof | — | — | — |
| Kineto | — | — | — |
