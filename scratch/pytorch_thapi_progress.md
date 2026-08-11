---
marp: true
title: "PyTorch → THAPI integration"
paginate: true
theme: default
---

<!--
Marp-style deck. Each `---` starts a new slide.
Render with: marp pytorch_thapi_progress.md -o out.pdf   (or --pptx / --html)
Or just read it as a normal Markdown document.
-->

# PyTorch → THAPI integration

**Tracing ATen operations into THAPI's CTF timeline — progress**

### Agenda

1. **PyTorch architecture** — the layered stack & its hook points
2. **Why we trace at the ATen dispatcher** — and not any other layer
3. **The ATen operator surface** — how many ops, public vs. private, composites & kernels
4. **Interposition & registration** — RecordFunction + LD_PRELOAD
5. **Producer design** — LTTng-UST tracepoints → CTF
6. **Progress** — Stage 0 → Stage 1 → Stage 2
7. **THAPI's pairing assumption** — no nested/reentrant API calls, and why ATen breaks it
8. **From CPU to GPU** — device-agnostic tracing
9. **CPU ↔ GPU correlation** — vs. Kineto, and why no PyTorch-side ID
10. **The deep-async test** — does containment survive without syncs?
11. **Open questions & next steps**

---

# PyTorch architecture — layers and tracing hook points

- **User code / nn.Module** — groups tensor operations into reusable model building blocks
- **torch.nn / functional** — the Python API where the user calls tensor operations (add, matmul, softmax…)
- **Autograd engine** — records each operation into a graph as it runs, later used to compute gradients
- **Dispatcher (ATen)** ⟵ *our hook* — for every operation, picks the right implementation based on device (CPU/GPU) and data type
- **C10 core** — defines the basic building blocks: Tensor, Storage, Device
- **Backend kernels** — the device-specific code that does the actual math

---

# Why trace at the ATen dispatcher?

| Layer | Hook | Why not |
|---|---|---|
| nn.Module | `nn.Module` hooks | sees model blocks, not individual ops; misses models written in raw ops |
| functional API | `__torch_function__` | Python-only — misses ops dispatched internally in C++ |
| Autograd | `autograd.Function` | gradient-graph nodes only; nothing under `no_grad` / inference |
| **Dispatcher (ATen)** | **`RecordFunction`** | ✅ **every op passes through — chokepoint** |
| C10 core | — | op identity already lost; no `aten::add` name |
| Backend kernels | CUPTI / XPUPTI | device layer — **THAPI already traces this** |

**Why the dispatcher wins**

- **Every operation passes through it** — one hook, whether called from Python or from inside C++
- **Lowest layer that still knows the op's name** (`aten::add`) — below it, that meaning is gone
- **Device-agnostic** — same hook fires for CPU and GPU
- **Fits THAPI's layering** — THAPI owns the bottom (Level Zero); we add the missing semantic layer on top

---

# The ATen operator surface — what we're tracing

- **Where they're defined** — all operators declared in `aten/src/ATen/native/native_functions.yaml` (a signature + backend dispatch table)

**How many**

- **2055** operator entries total
- **~557** are user-facing ("core")
- the rest: overloads (~840), in-place / out= variants, backward (78), internal ops (233)
- *(PyTorch dev-discuss, 2021)*

**Public vs. private**

- **Public** — variants: function, method → `torch.softmax(x)` ; plain names: add, matmul, softmax
- **Private / internal** — leading underscore: `_softmax, _local_scalar_dense` ; no Python binding — we don't want these

**Two behaviors that shape our trace**

- **Ops call other ops** — a composite op has no kernel of its own; it calls other ATen ops (`softmax → _softmax; linear → t + addmm`) → this is why the dispatch tree **nests**
- **Some ops launch no kernel** — view / metadata ops (`as_strided, unsqueeze, t`) only reshape tensor metadata

> **Takeaway:** we want the **public op the user called** — not the private, nested, kernel-less ops beneath it → *motivates Stage 2*

---

# Interposition & registration — RecordFunction + LD_PRELOAD

**◀ Interposition — intercept every op**
*One global callback pair at the dispatcher; fires around every ATen op, host-side.*

```cpp
std::unique_ptr<at::ObserverContext>
on_entry(const at::RecordFunction& fn) {
  tracepoint(lttng_ust_pytorch,
             op_entry, fn.name());
  return nullptr;
}

void on_exit(const at::RecordFunction& fn,
             at::ObserverContext*) {
  tracepoint(lttng_ust_pytorch,
             op_exit, fn.name());
}
```

**Registration — turn it on, untouched workload ▶**
*A constructor installs the callback at load time — before Python imports torch.*

```cpp
__attribute__((constructor))
void thapi_torch_init() {
  at::addGlobalCallback(
    at::RecordFunctionCallback(
        &on_entry, &on_exit)
      .scopes({at::RecordScope::FUNCTION}));
}
```

```bash
LD_PRELOAD=libthapi_torch.so \
    python3 model.py
```

> **OMPT analogy:** direct analog of OMPT's `OMP_TOOL_LIBRARIES` + `ompt_start_tool`

---

# Producer design — LTTng-UST tracepoints → CTF

- **No header extraction** — other THAPI backends parse API headers (`h2yaml`); PyTorch doesn't need it
- The entire ATen "API surface" collapses to **one generic op event**, op name carried as a string field
- Just **two tracepoints**: `op_entry(name)` / `op_exit(name)`
- Emitted to a **CTF trace** via LTTng-UST — same format every THAPI backend produces
- Per-event context at record time: `vpid / vtid`  (pairing & nesting key `{hostname, vpid, vtid}`)
- **Name-only, by design** — capturing args (shapes/dtypes) needs `.needsInputs(true)`, which boxes every argument into an `IValue` per op (measurable overhead); Stage 1/2 skip it

**The tracepoints (CTF event schema)**

```c
TRACEPOINT_EVENT(lttng_ust_pytorch,
    op_entry,
  TP_ARGS(const char *, name),
  TP_FIELDS(ctf_string(name, name)))

TRACEPOINT_EVENT(lttng_ust_pytorch,
    op_exit,
  TP_ARGS(const char *, name),
  TP_FIELDS(ctf_string(name, name)))
```

> Built with **`-DNDEBUG`** — without it the process crashes (root cause not yet confirmed)

---

# Progress — Stage 0 → Stage 1 → Stage 2

| Stage | What it added | Result |
|---|---|---|
| **Stage 0** | Interposition proof — one global callback, printed to stderr | confirmed every ATen op is interceptable |
| **Stage 1** | Real THAPI producer — LTTng-UST → CTF, activated by LD_PRELOAD | unmodified script → verified CTF trace |
| **Stage 2** | **Top-level-only filter** (depth == 0) — keep user-invoked ops, drop nested sub-ops | composites collapse to one event |

**Stage 1 vs. Stage 2 — same workload (`linear → softmax → sum`), real trace**

```
Stage 1 (all ops)              Stage 2 (top-level only)
aten::linear                   aten::linear
    aten::t                    aten::softmax
    aten::addmm                aten::sum
    aten::expand               aten::item
aten::softmax
    aten::_softmax
aten::sum
    aten::as_strided  ...
aten::item
    aten::_local_scalar_dense
```

> Depth tracked with a **thread-local counter** — emit only when depth returns to 0

---

# THAPI's pairing assumption — confirmed, and it's by design

**The property holds for every C-API interposition backend THAPI ships (`ze`, `cuda`, `hip`, `opencl`, `mpi`):** per thread, no traced API call enters another traced API call before it returns, and no reentrancy within the same call. THAPI's interval-matching engine doesn't just benefit from this — it **assumes** it.

**Where the assumption lives** — tracers emit raw `*_entry` / `*_exit` tracepoints; the *pairing* into intervals happens downstream in the `btx` filter:

```cpp
typedef std::tuple<hostname_t, process_id_t, thread_id_t> hpt_t;  // {hostname, vpid, vtid}

void    set_ts(hpt_t hpt, int64_t ts) { entry_ts[hpt] = ts; }     // entry: overwrite
int64_t get_ts(hpt_t hpt) { return thapi_at(entry_ts, hpt); }     // exit:  read (.at())
std::unordered_map<hpt_t, int64_t> entry_ts;                       // ONE slot per thread
```

**One timestamp slot per thread**, keyed only on `{hostname, vpid, vtid}` — no stack, no call-name, last-write-wins on entry. Correct only if per-thread calls strictly alternate entry → exit.

**Why it's desirable** — O(1), constant-memory pairing; simple matching model; matches the reality of flat driver/runtime APIs (`zeCommandListAppendLaunchKernel`, `clEnqueueNDRangeKernel`, `MPI_Send` are leaf calls that don't re-enter the traced surface).

**If the property is violated** — a nested inner entry overwrites the outer's timestamp; the inner exit consumes it; the outer exit then reads garbage → **negative/absurd durations**, mis-paired intervals, and a possible `.at()` **throw**. Silent corruption, not a clean error.

*Note:* CUDA's `in_init` / `_in_init` / `_in_init_cuda` guards protect tracer init / symbol loading only — **not** call nesting; and `cudart` vs the CUDA driver are two separate providers.

---

# THAPI's pairing assumption — contrast with PyTorch ATen ops

**ATen violates the property.** Composite ops call other ATen ops internally, on the same thread, within the same dispatch:

- `aten::linear` → `aten::t` + `aten::addmm`
- `aten::softmax` → `aten::_softmax`
- …any composite / decomposition

So `aten::*` calls **inherently nest on the same `{vpid, vtid}`**. Dropped into THAPI's single-slot `EntryState` unchanged, the inner op's entry clobbers the outer op's start → durations corrupt.

**Two things make ATen safe to trace anyway:**

1. **`at::RecordFunction` is built for nesting** — callbacks fire in proper **scoped LIFO** order (outer entry → inner entry → inner exit → outer exit), so the event stream is well-formed nesting, not garbage.
2. **Stage 2's `thread_local int g_depth` is the coping mechanism** — it counts nesting depth per thread, keeps pairing correct under nesting, and emits only **top-level (`depth == 0`) ops**, filtering out composite internals. Instead of *assuming* no nesting, we **measure** it.

> **Bottom line:** THAPI's single-slot matching is correct for the flat driver APIs it was designed for, and its cheapness is the payoff of that flatness. PyTorch ATen breaks the flat assumption structurally, so the PyTorch backend can't reuse the naive single-slot pairing as-is — it needs either the depth-gated top-level filter (Stage 2's `g_depth`) or a genuine per-thread **stack** in the matching model.

---

# From CPU to GPU — the tracer never changes

- **Device-agnostic** — `RecordFunction` fires the same on any `device`, CPU or GPU.
- **Same ops, different timings** — identical events, only durations change.
- **Host timeline, not kernels** — intervals measure dispatch/launch; big durations are host waits.
- **Sync ≈ no-sync** — traces match because the final `.item()` always waits for the GPU.
- **Limitation** — on-device kernel timing needs the layer below (Level Zero, THAPI's `ze`).

*Verified on Intel Data Center GPU Max 1550.*

---

# CPU ↔ GPU correlation — already solved one layer down

> **The problem:** `RecordFunction` times the **host launch** of an ATen op; the GPU kernel runs **async** and finishes later. Who times the kernel, and how is it tied back to the op?

| | Kineto / torch.profiler | THAPI |
|---|---|---|
| **Kernel timing** | XPUPTI / CUPTI activity records | injected Level Zero KERNEL_TIMESTAMP |
| **Correlation** | explicit correlation ID stamped on both streams | `{vpid, vtid}` + interval containment + device→host clock conversion |

→ **THAPI needs no Kineto/XPUPTI** — it already lives inside Level Zero, one layer below where kernels launch.

**Why we don't re-solve it**

```
aten::add
   our backend: naming, {vpid,vtid}
   │  contains
   ▼
zeAppendLaunchKernel
   ze backend already pairs this
   │  to real DEVICE start/end
   ▼
GPU runs kernel (async)
```

> So our layer needs no correlation ID of its own — containment carries it. But does that hold when PyTorch queues ops deep-async? → next slide

---

# The deep-async test — does containment survive without syncs?

**The worry**

- Correlation relies on each `aten::*` host interval **containing** its GPU launch.
- If PyTorch queues **many ops with no sync**, launches could stack up before any kernel runs — so nesting might not uniquely map kernel → op. This is where Kineto's explicit ID would look more robust.
- *So we tested it directly.*

**The experiment:** 50 ops chained, no intermediate sync (single trailing sync); big tensors so the host outruns the GPU. Both preloads in **one LTTng session**: our `aten::*` + THAPI's ze layer.

**What we found**

- **Containment held ~1:1** — 152 of 153 compute ops contained exactly one launch. *Why:* the launch submit returns in microseconds, **inside** the op's window — async delays when the kernel runs, not when it's appended. Only 1 op (trailing sum across the final sync) was ambiguous.
- **An explicit key exists anyway** — each launch's `hSignalEvent` matches the profiling record's `hEvent`: a real correlation ID living below PyTorch, independent of sync state.

> **Conclusion:** the PyTorch layer needs **no correlation ID of its own** — containment now, explicit `hSignalEvent↔hEvent` fallback from below if ever needed. Open question settled.

---

# Open questions & next steps

**Settled**

- ✅ **Where to hook** — ATen dispatcher via RecordFunction.
- ✅ **Device-agnostic** — one library, CPU + GPU.
- ✅ **CPU↔GPU correlation** — no PyTorch-side correlation ID needed.
- ✅ **Pairing safety** — ATen's nesting is handled by the Stage 2 depth counter, not THAPI's flat single-slot model.

**Open questions**

- **`-DNDEBUG` crash** — building against this libtorch without `-DNDEBUG` crashes the process.
- **Argument capture** — top-level ops currently carry **name only**. Shapes/dtypes need `.needsInputs(true)`, which boxes every arg into an `IValue` (measurable overhead). Question: which args, and how to present them.

**Next steps**

- → **Argument fields** on top-level ops (shapes/dtypes).
- → **THAPI integration** — fold the producer into THAPI's backend layout + a proper `btx` viewer/matching model (as the ze / omp backends have). *A per-thread stack (vs. single-slot EntryState) is the matching-model change ATen's nesting would require.*
