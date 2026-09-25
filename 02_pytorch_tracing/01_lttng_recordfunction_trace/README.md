# Step 1 — Generate an LTTng trace of unmodified PyTorch via the RecordFunction hook

**Goal of this step:** produce a real CTF trace of an *unmodified* PyTorch program by
injecting a RecordFunction callback through `LD_PRELOAD` and emitting LTTng-UST
tracepoints. RecordFunction is the **hook**; `LD_PRELOAD` is the **delivery**; LTTng is
the **backend** that stamps time + thread and writes the trace.

No Python code is changed — `example/model.py` has no idea a tracer exists.

## Contents

```
env.sh                        # module recipe (source before anything)
example/model.py              # the unmodified workload (Linear -> relu)
tracer/tracer.cpp             # the RecordFunction tracer (entry/exit -> tracepoint)
tracer/pytorch_tracepoints.tp # LTTng-UST provider schema (op_entry / op_exit)
tracer/build.sh               # build libtorch_tracer.so
tracer/run.sh                 # open session + LD_PRELOAD run + record
tracer/view.sh                # babeltrace2 dump
trace_output.txt              # the exact 268-event trace produced (see below)
```

## How to reproduce (module-based, no absolute paths)

```bash
source env.sh lttng          # loads oneapi/release/2025.3.1 -> lttng-tools -> babeltrace2 -> frameworks (LAST)
tracer/build.sh              # -> tracer/libtorch_tracer.so
tracer/run.sh                # LD_PRELOAD on example/model.py, records CTF
tracer/view.sh               # babeltrace2 dump of the newest session
```

## Where the trace is written

Read the trace with either:

```bash
source env.sh lttng
tracer/view.sh                                                    # auto-picks newest session
# or directly:
babeltrace2 $HOME/lttng-traces/thapi-pytorch-session-20260819-135516
```

## The tracer (minimal)

Entry/exit just fire a name-only tracepoint; LTTng stamps time + vpid/vtid, so no manual
clock, tid, or correlation id is needed:

```cpp
std::unique_ptr<at::ObserverContext> on_entry(const at::RecordFunction& fn) {
  tracepoint(lttng_ust_pytorch, op_entry, fn.name());
  return nullptr;
}
void on_exit(const at::RecordFunction& fn, at::ObserverContext*) {
  tracepoint(lttng_ust_pytorch, op_exit, fn.name());
}
__attribute__((constructor)) void tracer_init() {
  at::addGlobalCallback(at::RecordFunctionCallback(&on_entry, &on_exit)
      .scopes({at::RecordScope::FUNCTION}));
}
```

## The trace (268 events)

The `torch.relu(self.linear(x))` region, indented to show the nesting (LIFO entry/exit):

```
op_entry aten::linear          <- composite op, outermost
  op_entry aten::t
    op_entry aten::transpose
      op_entry aten::as_strided
      op_exit  aten::as_strided
    op_exit  aten::transpose
  op_exit  aten::t
  op_entry aten::addmm         <- the real GEMM (Wx+b)
    op_entry aten::expand
    op_entry aten::copy_
    op_entry aten::resolve_conj
  op_exit  aten::addmm         (+0.009 s)
op_exit  aten::linear
op_entry aten::relu            <- outermost
  op_entry aten::clamp_min     <- relu's real kernel
  op_exit  aten::clamp_min
op_exit  aten::relu
```

Each real event line from babeltrace2 looks like:

```
[13:55:18.878163532] aurora-uan-0010 lttng_ust_pytorch:op_entry: { cpu_id = 36 }, { vpid = 27831, vtid = 27831 }, { name = "aten::linear" }
```

The complete dump is in `trace_output.txt`.

## What this demonstrates

- **LTTng stamps time + vpid/vtid**, so the callback stays trivial (no manual clock/tid/id).
  Pairing is geometric (time + thread) — the THAPI model.
- **Nesting is visible** via LIFO entry/exit ordering: `aten::linear` -> `aten::t`/`aten::addmm`
  -> their sub-ops.
- **This is the outermost view — and that follows from *where* the hook sits in the dispatcher.**
  `aten::linear` shows its decomposition into other *named ATen ops* (`aten::t`, `aten::addmm`, ...)
  but not per-dispatch-key redispatch hops (Autograd -> CPU). Here is why:

  The RecordFunction bracket is placed **once, at the top of the op**, so what we see is the op
  *and the other named ATen ops it calls*: those inner ops each re-enter `Dispatcher::call` and
  re-arm the observer gate, so they get their own `op_entry`/`op_exit` bracket. What we do **not**
  see is the same op being re-dispatched across dispatch keys (Autograd -> CPU): the dispatcher's
  `redispatch`/`redispatchBoxed` paths are **intentionally not instrumented**
  (`Dispatcher.h:840, 898` — "do not use RecordFunction on redispatch"), so a single op fires our
  callback exactly once regardless of how many key-to-key hops it takes internally. Reaching that
  finer granularity needs a different mechanism (e.g. `TorchDispatchMode` in Python, or raw-symbol
  interposition of `KernelFunction::call`/`lookup`, which sees every hop) — explored in a later step.
