# 2026_TracingIAML — assets

This folder holds the `2026_TracingIAML.md` presentation plus the trace
files and rendered timeline images it embeds, for the workload:

```python
import torch
with torch.autograd.profiler.emit_itt():
    x = torch.randn(2048, 2048, device="xpu")
    y = torch.matmul(x, x)
torch.xpu.synchronize()
```

## Files

- `2026_TracingIAML.md` — the slides
- `iprof_timeline.pftrace` / `kineto_timeline.json` — raw traces (Perfetto-compatible)
- `iprof_timeline.png` / `kineto_timeline.png` / `vtune_timeline.png` — rendered Gantt-style timeline images embedded in the slides
- `render_timelines.py` — script that produced the three PNGs

## Perfetto links

The slides link `iprof_timeline.pftrace` and `kineto_timeline.json` to
`ui.perfetto.dev` via:
```
https://ui.perfetto.dev/#!/?url=https://raw.githubusercontent.com/DonAurelio/notes/main/slides/2026_TracingIAML/<file>
```
This only works once the repo is pushed (public), since `raw.githubusercontent.com`
must actually serve the file for Perfetto's `fetch()` to load it.

VTune has **no Perfetto-compatible export** — its result directory
(`vtune_xpu_offload/`, not included here) is a proprietary binary format.
View it natively instead:
```bash
vtune-gui <result-dir>
# or, browser-based, no X-forwarding needed:
vtune-backend --data-directory=<parent-of-result-dir>
```

## Regenerating the timeline images

```bash
source ~/.claude/skills/pytorch-basic-tracing-environment-config/scripts/env.sh lttng
cd 2026_TracingIAML
python render_timelines.py
```

`render_iprof()` and `render_kineto()` read real per-event start/duration
timestamps straight out of `iprof_timeline.pftrace` (via the `perfetto`
Python trace-processor package) and `kineto_timeline.json` respectively.

### VTune timeline — how the data was recovered

VTune's CLI has **no report that dumps raw per-event start times** — `-report
timeline` only emits coarse, equal-width time bins (`Bin Start/End Time` +
aggregated metric), and the result directory's internal `sqlite-db/` is a
proprietary binary layout, not meant to be queried directly.

Instead, each task/kernel's start time was recovered by **bisecting
`-time-filter`**: for a task with total duration `D` (already known from
`vtune -report hotspots -group-by task|computing-task`), the narrowest
window `[t, ELAPSED]` that still reports that task's *full* duration `D`
gives `t ≈` the task's actual start time — querying a window that starts
even slightly after the task began truncates its reported time below `D`.

```bash
# One data point, e.g. aten::matmul (full duration already known: 0.065117s):
vtune -report hotspots -result-dir vtune_xpu_offload -group-by task \
  -time-filter <t>:207.918 -format csv
# bisect <t> until "aten::matmul" time in the output stops matching 0.065117
```

This was automated (see `bisect_vtune.py`-style logic, not kept in this repo
since it was a one-off ~30-iteration binary search per task, each iteration
being a full `vtune -report` call — a few minutes total for 6 data points).
Run it with the system `vtune` directly (`/opt/.../oneapi/vtune/latest/bin64/vtune`,
**not** the one on `PATH` after sourcing the PyTorch tracing `env.sh`/`lttng`
module profile — that profile resolves to an older VTune install, and opening
a 2026.4.0-collected result with it upgrades the result's database schema
irreversibly, after which the newer `vtune` can read it fine but the result
should not be touched with the older one again).

Recovered values (seconds, from `vtune_xpu_offload`, elapsed time 207.918s):

| Task | Start (s) | Duration (s) |
|---|---|---|
| `aten::randn` | 207.706601 | 0.006225 |
| `aten::normal_` | 207.707071 | 0.005750 |
| `aten::matmul` | 207.712904 | 0.065117 |
| `aten::mm` | 207.712926 | 0.065091 |
| `DistributionElementwiseKernelFunctor...` (GPU) | 207.712828 | 0.000043 |
| `gemm_kernel` (GPU) | 207.777999 | 0.000820 |

These are hard-coded in `render_timelines.py`'s `render_vtune()` — re-run the
bisection against a fresh `vtune_xpu_offload` result if the experiment is
repeated and the numbers need updating.
