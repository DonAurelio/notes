# iprof vs Kineto — Tracing Overhead Experiment (exp_v1.2)

**Date**: 2026-10-26  
**Platform**: Aurora (Intel XPU, 1 node)  
**Workload**: `torch.matmul` — sizes S ∈ {256, 1024, 4096}, iterations N ∈ {10, 100, 1000}  
**Reps**: 15 per cell · **Warmup**: 5 iterations (excluded from timing)  
**Scenarios**: BL-XPU (no tracer), IP-XPU (iprof/THAPI), KN-XPU (Kineto)

---

## 1. Event counting methodology

A key correction from exp_v1.0: both tracers had been counted by tally intervals
(matched entry/exit pairs), not raw trace records.

| Tool | Recording model | n_events definition (exp_v1.2) |
|---|---|---|
| iprof (THAPI) | 2 LTTng events per call (entry + exit) + fixed setup metadata | `babeltrace2 <raw_trace> \| wc -l` inline after each collection |
| Kineto | 1 FunctionEvent per invocation (entry sets start_ns, exit sets end_ns on same object) | `sum(e.count for e in prof.key_averages())` — already correct, no ×2 |

Formula correction (2 × intervals + metadata(S,N)) was rejected because metadata event
count varies across (S, N) due to JIT compilation and kernel specialization. Direct
babeltrace2 count per rep is the only reliable method.

---

## 2. T_hot overhead

T_hot is the wall time of the hot loop (warmup excluded). Overhead is reported as
(T_hot_traced − T_hot_BL) / T_hot_BL × 100%.

| S | N | BL mean ± std (s) | iprof mean ± std (s) | iprof overhead | Kineto mean ± std (s) | Kineto overhead |
|---|---|---|---|---|---|---|
| 256 | 10 | 0.000298 ± 0.000005 | 0.000422 ± 0.000007 | **+41.9%** | 0.000711 ± 0.000012 | **+138.9%** |
| 256 | 100 | 0.002113 ± 0.000027 | 0.003448 ± 0.000027 | **+63.2%** | 0.005843 ± 0.000116 | **+176.5%** |
| 256 | 1000 | 0.019773 ± 0.000169 | 0.033114 ± 0.000187 | **+67.5%** | 0.058890 ± 0.000679 | **+197.8%** |
| 1024 | 10 | 0.001156 ± 0.000002 | 0.001210 ± 0.000007 | +4.7% | 0.001266 ± 0.000008 | +9.5% |
| 1024 | 100 | 0.011094 ± 0.000003 | 0.011511 ± 0.000006 | +3.8% | 0.011672 ± 0.000013 | +5.2% |
| 1024 | 1000 | 0.111501 ± 0.000121 | 0.114718 ± 0.000074 | +2.9% | 0.116336 ± 0.000091 | +4.3% |
| 4096 | 10 | 0.064577 ± 0.000073 | 0.064611 ± 0.000067 | ~0% | 0.064637 ± 0.000057 | ~0% |
| 4096 | 100 | 0.691270 ± 0.000640 | 0.691210 ± 0.000325 | ~0% | 0.690923 ± 0.000596 | ~0% |
| 4096 | 1000 | 7.215276 ± 0.011926 | 7.223067 ± 0.011569 | ~0% | 7.229136 ± 0.007675 | ~0% |

**Key finding**: overhead is strongly workload-dependent. At S=256 (small matrix, short
hot loop), both tracers impose substantial per-iteration cost: iprof up to +67%,
Kineto up to +198%. At S=1024 overhead falls below 10% for both. At S=4096 (hot loop
≥ 65 ms per iteration) tracing cost is within measurement noise for both tools.

![Overhead ratio](overhead_ratio.png)

---

## 3. Process startup cost

T_process is total wall time including Python and tracer initialization (warmup included).
The difference (T_process − T_hot) captures tracer startup cost.

| Tool | T_process range (s) | Startup overhead vs BL |
|---|---|---|
| BL-XPU | 2.0 – 9.4 | — |
| IP-XPU | 10.5 – 18.0 | **+8.5 – 9.5 s** (LTTng session setup) |
| KN-XPU | 3.1 – 12.5 | **+1.0 – 3.1 s** (Kineto initialization) |

iprof carries a fixed ~9 s startup penalty from LTTng session creation regardless of
workload size. Kineto startup is significantly cheaper and scales slightly with N
(larger profiling buffers at higher iteration counts).

---

## 4. Event counts

n_events is deterministic across reps (std = 0 for all cells).

| Tool | S | N=10 | N=100 | N=1000 | events/matmul |
|---|---|---|---|---|---|
| iprof | 256 | 3,805 | 5,515 | 22,627 | ~19 |
| iprof | 1024 | 3,808 | 5,338 | 20,644 | ~17 |
| iprof | 4096 | 3,834 | 5,364 | 20,670 | ~17 |
| Kineto | 256 | 1,617 | 4,497 | 33,300 | ~32 |
| Kineto | 1024 | 1,755 | 5,355 | 41,358 | ~40 |
| Kineto | 4096 | 1,799 | 5,579 | 43,382 | ~42 |

iprof starts with higher fixed event count (~3,800 at N=10, dominated by ZE API
initialization calls) but lower per-iteration growth (~17–19 events/matmul). Kineto
starts lower (~1,600–1,800) but grows faster (~32–42 events/matmul). The crossover
occurs around N=100 for S=256 and between N=100 and N=1000 for larger matrices.

![Event count comparison](n_events_comparison.png)

---

## 5. Analysis cost

analysis_s is the time to post-process the trace (`iprof --replay` for iprof,
`prof.key_averages()` for Kineto).

| Tool | S | N=10 | N=100 | N=1000 | Scaling |
|---|---|---|---|---|---|
| iprof | 256 | 0.723 ± 0.401 s | 0.606 ± 0.010 s | 0.627 ± 0.018 s | flat |
| iprof | 1024 | 0.608 ± 0.049 s | 0.621 ± 0.055 s | 0.620 ± 0.012 s | flat |
| iprof | 4096 | 0.594 ± 0.008 s | 0.609 ± 0.010 s | 0.643 ± 0.055 s | flat |
| Kineto | 256 | 0.051 ± 0.001 s | 0.142 ± 0.001 s | 1.315 ± 0.005 s | O(n) |
| Kineto | 1024 | 0.055 ± 0.001 s | 0.169 ± 0.002 s | 1.711 ± 0.017 s | O(n) |
| Kineto | 4096 | 0.057 ± 0.001 s | 0.175 ± 0.002 s | 1.787 ± 0.023 s | O(n) |

iprof analysis is dominated by a fixed babeltrace2 cost (~0.6 s) regardless of event
count. Analysis cost per event therefore drops from ~190 μs/event at N=10 to ~28 μs/event
at N=1000 as the fixed cost amortizes.

Kineto analysis scales linearly with event count at a near-constant ~31–41 μs/event
across all cells, consistent with pure O(n) aggregation.

At N=1000, Kineto analysis takes 2–3× longer than iprof despite a larger event count,
because iprof's flat cost saturates well below Kineto's linear trajectory.

![Analysis cost per event](analysis_per_event.png)

---

## 6. Summary

| Dimension | iprof (THAPI) | Kineto |
|---|---|---|
| T_hot overhead (small S) | Up to +67% | Up to +198% |
| T_hot overhead (large S) | ~0% | ~0% |
| Startup cost | ~9 s (LTTng, fixed) | ~1–3 s |
| Events/matmul | ~17–19 | ~32–42 |
| Analysis scaling | Flat ~0.6 s | O(n), ~31–41 μs/event |
| Analysis cost at N=1000 | ~0.6 s | ~1.3–1.8 s |

For memory-bandwidth-bound workloads (small matrices, short hot loops), iprof imposes
lower per-iteration tracing overhead than Kineto. For compute-bound workloads (large
matrices), both tracers are negligible. iprof carries a higher one-time startup cost
but provides cheaper post-run analysis at scale. Kineto offers faster startup and
lower per-event analysis cost in the low-N regime.
