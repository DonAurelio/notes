# CPU path: a real VTune result from this system

**PyTorch version:** `2.10.0a0+git449b176` (commit
`449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`). VTune version `2026.4.0`
(`vtune --version`, binary at
`/opt/aurora/26.181.0/oneapi/vtune/latest/bin64/vtune`, already on `PATH`,
no module load needed).

**How this was produced:** `example/profile_itt.py` runs 30,000 iterations
of `MyModel`'s forward+backward inside `emit_itt(record_shapes=True)`
(see `docs/00-overview.md` for what that call does internally), launched
under VTune on the Aurora login node:

```bash
source env.sh
PYBIN=$(which python3)   # vtune's `--` does NOT inherit the sourced
                          # module PATH the way a plain shell does —
                          # a bare `python3` resolves to /usr/bin/python3
                          # (system Python 3.6, no torch) under vtune --
cd example
vtune -collect hotspots -result-dir ../results/vtune_hotspots -- "$PYBIN" profile_itt.py
```

Full raw output saved at `results/vtune_hotspots/console_log.txt`; the
finalized report at `results/vtune_hotspots/summary.txt`. The result
directory itself (`results/vtune_hotspots/`) is ~588 MB — too large to
commit to git; only the two text files above are meant to travel with this
repo.

## The real "Top Tasks" table

```
$ vtune -report summary -r results/vtune_hotspots -format=text -report-knob show-issues=false
...
Top Tasks
Task Type                                            Task Time  Task Count  Average Task Time
---------------------------------------------------  ---------  ----------  -----------------
autograd::engine::evaluate_function: AddmmBackward0     0.932s      30,000             0.000s
aten::linear                                            0.816s      30,000             0.000s
aten::sum                                               0.644s      90,000             0.000s
aten::t                                                 0.569s     120,000             0.000s
aten::addmm                                             0.551s      30,000             0.000s
[Others]                                                4.465s   1,320,000             0.000s
...
Result Size: 588.0 MB
Collection start time: 21:07:03 02/10/2026 UTC
Collection stop time: 21:12:27 02/10/2026 UTC
```

This is the direct evidence for `docs/00-overview.md`'s claim that
`pushITTCallbacks` installs one RecordFunction callback whose start/end
handlers call `itt_range_push`/`itt_range_pop` — **every row in VTune's
"Top Tasks" table is literally the `fn.name()` string passed to
`itt_range_push`, i.e. the exact same op name RecordFunction gives every
other observer** (Kineto, this project's LTTng tracer). There is no
ITT-specific naming or transformation anywhere in the pipeline.

## Reconciling the task counts against the loop

`profile_itt.py`'s loop body is:

```python
for _ in range(ITERS):          # ITERS = 30000
    output = model(x)
    loss = output.sum()
    loss.backward()
    model.zero_grad()
```

`aten::addmm`/`aten::linear` (both `30,000` = exactly 1× per iteration) and
`autograd::engine::evaluate_function: AddmmBackward0` (also `30,000`, 1×)
match a direct call-count check with `torch.autograd.profiler_legacy.profile`
(the CPU-only legacy profiler, used here only to enumerate op names/counts
per iteration — not part of the ITT path itself) run over the identical
loop body:

```
$ python3 -c "... with legacy_profile() as prof: <loop body> x3 ...
               Counter(e.name for e in prof.function_events)"
aten::sum 6   (2.0 / iteration)
aten::t   12  (4.0 / iteration)
aten::addmm 3 (1.0 / iteration)
aten::linear 3 (1.0 / iteration)
```

`aten::t`'s VTune count (`120,000` = exactly 4× `30,000`) matches this
check exactly. **`aten::sum`'s VTune count (`90,000` = 3× `30,000`) does
not match** the legacy-profiler count of 2×/iteration from a standalone
re-run of the same loop body — this discrepancy is reported here as
observed, not resolved; a likely cause is some interaction between VTune's
own sampling/task-aggregation logic and nested/async autograd calls that
the standalone legacy-profiler check doesn't reproduce exactly, but that
is a hypothesis, not a confirmed explanation, and is flagged as an open
question rather than asserted as fact.

## What this confirms vs. what Kineto's equivalent section shows

Compare to `../../kineto-deep-dive/docs/02-cpu-path.md`: that doc annotates
one exported **JSON event object** (`ts`, `dur`, `args.*` fields) against
PyTorch's internal `Result`/`ExtraFields<TorchOp>` C++ structs, because
Kineto builds and exports that structured event itself. Here, there is no
analogous "exported event object" to annotate — VTune's "Top Tasks" table
is VTune's *own* aggregation of raw `__itt_task_begin`/`__itt_task_end`
pairs it sampled from the ittnotify ring buffer; PyTorch contributes
nothing but the task *name* at each push/pop call. This is the concrete,
measured illustration of `docs/00-overview.md` §1's point: ITT's "export"
step is just "a separate process was already watching."

## What's next

- `docs/02-no-device-support.md` — the live XPU validation run (same
  script, `DEVICE=xpu`, run on a reserved Aurora compute node) and the
  explicit, evidence-backed answer on CUDA/XPU-device support.
