# Step 2 — RecordFunction trace granularities

Step 1 proved the seam works: a C++ `RecordFunction` callback, built as a `.so`,
`LD_PRELOAD`ed into an **unmodified** `model.py`, emits LTTng events for every
dispatched op. This step asks the next question: **what can we turn up or down in
that trace, and what does each knob cost or reveal?**

One `.so` covers every experiment. Each granularity is a load-time toggle read
from an environment variable, so the *same build* produces every trace below —
you flip a variable, not recompile.

| Aspect | Env toggle | Values | What it changes |
|---|---|---|---|
| 2 — scope | `TRACER_SCOPES` | `function` \| `function+backward` | which `RecordScope`s fire the callback |
| 3 — thread | `TRACER_THREAD` | `global` \| `local` | callback reaches every thread vs the registering thread only |
| 4 — depth | `TRACER_TOPLEVEL` | `0` \| `1` | emit all nested ops vs only user-level (`depth==0`) ops |
| 5 — inputs | `TRACER_INPUTS` | `0` \| `1` | attach rendered `dtype[shape]@device` args (`.needsInputs`) |
| 6 — sampling | `TRACER_SAMPLING` | `0.0`–`1.0` | fraction of ops sampled (global callbacks only) |

(Aspect 1, "every op fires one entry/exit pair", is the Step 1 baseline. Aspect
6' — per-dispatch-key granularity — is **not reachable** through this hook; see
the note at the end.)

Every event *always* carries `name`, `scope`, and `depth`, so aspects 2 and 4 are
visible directly in the data; the toggles change *what is captured* to isolate
each granularity's effect. Aspect 3 is read from LTTng's `vtid` context.

---

## Contents

```
02_recordfunction_granularities/
├── README.md                  # this file
├── env.sh                     # module recipe (same as Step 1)
├── example/
│   ├── model.py               # Linear(10,5) -> relu, forward only
│   └── model_backward.py      # + loss.backward()  (produces autograd nodes)
├── tracer/
│   ├── tracer.cpp             # the callback, all toggles, heavily commented
│   ├── pytorch_tracepoints.tp # LTTng-UST schema: name + scope + depth + args
│   ├── build.sh               # lttng-gen-tp + compile -> libtorch_tracer.so
│   ├── run.sh                 # run.sh <label> <workload.py>  (toggles via env)
│   └── view.sh                # babeltrace2 <trace-dir>
└── traces/                    # babeltrace2 dumps, one per experiment
    ├── baseline.txt           # forward, default toggles            (268 events)
    ├── scope_fwd.txt          # FUNCTION only, backward workload    (122)
    ├── scope_bwd.txt          # FUNCTION+BACKWARD                   (134)
    ├── thread_global.txt      # global callback                     (134)
    ├── thread_local.txt       # thread-local callback               (134)
    ├── depth_top.txt          # TOPLEVEL=1                          (92)
    ├── inputs.txt             # needsInputs                         (268)
    ├── sample_100.txt         # sampling = 1.0                      (268)
    ├── sample_050.txt         # sampling = 0.5                      (116)
    └── sample_010.txt         # sampling = 0.1                      (16)
```

---

## How to reproduce (module-based, no absolute paths)

```bash
source env.sh lttng                 # loads oneapi 2025.3.1, lttng, babeltrace2, frameworks LAST
cd tracer && ./build.sh             # -> libtorch_tracer.so

# baseline
./run.sh baseline ../example/model.py

# aspect 2 — scope
./run.sh scope_fwd ../example/model_backward.py
TRACER_SCOPES=function+backward ./run.sh scope_bwd ../example/model_backward.py

# aspect 3 — thread
TRACER_SCOPES=function+backward TRACER_THREAD=global ./run.sh thread_global ../example/model_backward.py
TRACER_SCOPES=function+backward TRACER_THREAD=local  ./run.sh thread_local  ../example/model_backward.py

# aspect 4 — depth
TRACER_TOPLEVEL=1 ./run.sh depth_top ../example/model.py

# aspect 5 — inputs
TRACER_INPUTS=1 ./run.sh inputs ../example/model.py

# aspect 6 — sampling
TRACER_SAMPLING=1.0 ./run.sh sample_100 ../example/model.py
TRACER_SAMPLING=0.5 ./run.sh sample_050 ../example/model.py
TRACER_SAMPLING=0.1 ./run.sh sample_010 ../example/model.py
```

---

## Where the traces are written

LTTng writes raw CTF to `$HOME/lttng-traces/step3-<label>-<timestamp>/` (each
experiment gets its own `step3-<label>` session). Read the latest for a label with:

```bash
source env.sh lttng
./tracer/view.sh "$(ls -dt $HOME/lttng-traces/step3-baseline-* | head -1)"
# module used to read: babeltrace2/2.1.2-archive (pulled in by `source env.sh lttng`)
```

The `traces/*.txt` files here are those babeltrace2 dumps, saved so the results
are readable without rerunning.

---

## The results

### Aspect 2 — scope: `TRACER_SCOPES`
`RecordFunctionCallback.scopes({...})` selects which `RecordScope`s invoke the
callback. Registering `FUNCTION` only vs `FUNCTION + BACKWARD_FUNCTION` on the
same `model_backward.py`:

```
scope_fwd (FUNCTION only):        122 FUNCTION
scope_bwd (FUNCTION+BACKWARD):     122 FUNCTION + 12 BACKWARD_FUNCTION
```

The 12 extra events *are* the autograd graph — the backward nodes PyTorch runs to
compute gradients:

```
AddmmBackward0   ReluBackward0   SumBackward0   TBackward0   torch::autograd::AccumulateGrad
```

Forward-only tracing never sees them; you must opt into `BACKWARD_FUNCTION`.

### Aspect 3 — thread: `TRACER_THREAD`
`addGlobalCallback` installs on every thread; `addThreadLocalCallback` installs
only on the thread that registers it. **On CPU the two traces are identical (134
events each, same single `vtid`)** — because the autograd engine runs the backward
pass *inline on the calling thread* here, so a thread-local callback still sees
it. The global-vs-local distinction only becomes observable when backward (or
data-loading, or a device runtime) executes on a *separate* worker thread; there,
thread-local silently misses it and global is required. This is the concrete
reason a general tracer uses `addGlobalCallback`.

### Aspect 4 — depth: `TRACER_TOPLEVEL`
A `thread_local` counter records nesting depth on every event. `depth==0` is a
user-level op; `depth>0` is an op the dispatcher fanned out to internally
(e.g. `aten::detach` -> inner `detach`). Emitting only `depth==0`:

```
baseline (all depths):   268 events
depth_top (depth==0):     92 events
```

Same run, ~3× fewer events — the trace collapses to the ops a user actually wrote,
hiding the internal dispatch subtree.

### Aspect 5 — inputs: `TRACER_INPUTS`
`.needsInputs(true)` makes the dispatcher box the op arguments so `fn.inputs()` is
populated in the start callback; the tracer renders each as `dtype[shape]@device`:

```
aten::detach     args="Float[5,10]@cpu"
aten::uniform_   args="Float[5,10]@cpu, scalar, scalar, None"
aten::empty      args="list, None, None, Device, scalar, None"
```

This is the richest granularity and the most expensive: it forces argument boxing
on the hot path, so it is off by default. (It also requires the `.so` be built
`-DNDEBUG` — see the ABI note in Step 1 — or `inputs()` trips a debug assert.)

### Aspect 6 — sampling: `TRACER_SAMPLING`
`.samplingProb(p)` tells the dispatcher to fire the callback on only ~`p` of ops.
Entry *and* exit are skipped together, so nesting depth stays balanced. Sweeping
`p` on the same forward run:

```
p = 1.0   268 events
p = 0.5   116 events
p = 0.1    16 events
```

The survivors are a *random subset* of ops (e.g. at `p=0.1`: `aten::max`,
`aten::fill_`, `aten::to`, `aten::as_strided`, …), not the first N — sampling
trades completeness for overhead, useful on long/hot workloads where a full trace
is too large. **Caveat:** sampling applies to **global** callbacks only;
thread-local callbacks always run regardless of `p`.

---

## What this demonstrates

**Per-dispatch-key granularity is NOT reachable here.**
RecordFunction sees an op **once, at its outermost dispatch**. It exposes
`name()`, `scope()`, `seqNr()`, `inputs()` — but **no `DispatchKey` accessor**, and
the dispatcher deliberately does **not** re-arm RecordFunction on `redispatch` /
`redispatchBoxed` (`Dispatcher.h:840, 898`: *"do not use RecordFunction on
redispatch"*). So the key-by-key hops an op takes internally
(Autograd → CPU → …) are invisible through this hook: you see `aten::addmm` one
time, not each dispatch-key stage of it. Reaching that level requires a different
seam (raw-symbol interposition of `KernelFunction::call`/lookup, or a Python
`TorchDispatchMode`). Documenting *why* is the deep-dive for the next step.
