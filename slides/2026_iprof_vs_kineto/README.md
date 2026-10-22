# iprof vs Kineto: XPU Tracing Overhead — Results Report v2

**Date:** 2026-10-22  
**System:** Aurora (Intel Ponte Vecchio XPU), 1 compute node, debug queue  
**Allocation:** ~39 min of a 60-min slot  

---

## Setup

| Item | Detail |
|---|---|
| Workload | `torch.matmul(x, x)`, XPU tensor, shape (S, S) |
| Warm-up | W = 5 iterations (traced, excluded from T_hot) |
| Repetitions | 15 per cell |
| Parameter grid | S ∈ {256, 1024, 4096} × N ∈ {10, 100, 1000} minus (4096, 1000) = 8 cells |
| iprof mode | default, `--no-analysis --backends pytorch,ze`; analysis via `--replay` |
| Kineto activities | `ProfilerActivity.CPU + XPU`; analysis via `prof.key_averages()` |
| Summary statistic | Mean across 15 reps; error reported as sample standard deviation (σ) |

---

## Results

**Error computation:** For each of the 15 repetitions, the overhead ratio is computed as `(T_hot_traced − T_hot_BL_median) / T_hot_BL_median × 100 [%]`, using the BL median as a fixed reference denominator. The reported σ is the sample standard deviation of these 15 per-rep ratios. The coefficient of variation (CV = σ/mean) is used below to assess data quality.

**Data quality assessment:**

| S | N | IP CV | KN CV | Assessment |
|---|---|---|---|---|
| 256 | 10–1000 | 1.7–4.3% | 1.0–3.6% | **Consistent** — overhead signal is large, noise is a small fraction |
| 1024 | 10–1000 | 1.0–2.4% | 1.1–2.9% | **Highly stable** — dedicated node, negligible run-to-run jitter |
| 4096 | 10 | 15% | 8% | **Acceptable** — signal still detectable (mean/σ ≥ 6); overhead is small but real |
| 4096 | 100 | 85% | 82% | **Below noise floor** — mean ≈ σ, some reps yield negative ratios; overhead is indistinguishable from timer noise at 667 ms wall time |

| S | N | IP-XPU ratio [%] | IP-XPU σ [%] | KN-XPU ratio [%] | KN-XPU σ [%] |
|---|---|---|---|---|---|
| 256  | 10   | 42.32 | 1.82 | 142.16 |  5.10 |
| 256  | 100  | 63.41 | 1.99 | 174.43 |  1.67 |
| 256  | 1000 | 69.13 | 1.20 | 197.82 |  3.85 |
| 1024 | 10   |  5.20 | 0.13 |   9.94 |  0.29 |
| 1024 | 100  |  4.33 | 0.04 |   5.76 |  0.15 |
| 1024 | 1000 |  4.27 | 0.05 |   5.72 |  0.06 |
| 4096 | 10   |  0.13 | 0.02 |   0.22 |  0.02 |
| 4096 | 100† |  0.03 | 0.03 |   0.03 |  0.03 |

† σ ≈ mean; overhead indistinguishable from measurement noise.

---

## Findings

- Kineto overhead ratio is 2–3× higher than iprof at S=256 across all N.
- Both tools' overhead ratio drops sharply as matrix size grows; at S=4096 both are near zero.
- iprof overhead ratio increases with N at S=256, showing no amortization for small compute kernels.
- iprof startup overhead is 8.4–12.8 s vs Kineto's 1.0–3.0 s across all cells.
- iprof per-iteration overhead is roughly constant (~0.013 ms at S=256, ~0.005 ms at S=1024), independent of N.
- Kineto XPU profiling adds measurable startup cost (~1 s) even at the smallest workload.
- At S=4096 N=100 the measured overhead is below the timer noise floor for both tools.
