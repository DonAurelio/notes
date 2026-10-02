# Event model: classification, storage, and whether the format changes per activity

**PyTorch version:** `2.10.0a0+git449b176` (commit
`449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`). Citations are `path:line` into:

```
$TORCH_DIR = /opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/lib/python3.12/site-packages/torch
```

Builds on `docs/00-overview.md` §4 ("the two-buffer model") — read that
first if you haven't. This doc answers: what kinds of events exist, where
each one lives before export, and whether the on-disk/in-memory shape
differs between CPU, XPU, and CUDA.

## 1. `EventType` — the master classification

Every event PyTorch's own collector (not Kineto's device plugin) records is
tagged with one `EventType`:

```cpp
// torch/csrc/profiler/collection.h:30-41
enum class EventType : uint8_t {
  TorchOp = 0,
  Backend,
  Vulkan,
  Allocation,
  OutOfMemory,
  PyCall,
  PyCCall,
  Kineto,
  PythonGC
};
```

Grouped by what the user's original question called "host, memory, or
other":

| `EventType` | category | struct | struct location |
|---|---|---|---|
| `TorchOp` | host (op execution) | `ExtraFields<TorchOp> : TorchOpBasicFields` | `collection.h:116-127` (basic fields), `141-183` (extra fields) |
| `Backend` | host (non-ATen backend span, e.g. JIT/lite-interpreter) | `ExtraFields<Backend>` | `collection.h:186-194` |
| `Vulkan` | host (Vulkan compute op) | `ExtraFields<Vulkan>` | `collection.h:203-211` |
| `Allocation` | **memory** (tensor alloc/dealloc) | `ExtraFields<Allocation> : RawAllocation` | `collection.h:213-221` (`RawAllocation`), `229-238` (`ExtraFields`) |
| `OutOfMemory` | **memory** (allocator OOM) | `ExtraFields<OutOfMemory>` | `collection.h:241-253` |
| `PyCall` / `PyCCall` | host (Python call stack) | `ExtraFields<PyCall>` / `<PyCCall>` | `collection.h:255-354` region |
| `Kineto` | **device bridge** (see `docs/00-overview.md` §4) | `ExtraFields<Kineto>` | `collection.h:356-373` |
| `PythonGC` | host (Python garbage collection pause) | `ExtraFields<PythonGC>` | `collection.h:197-200` |

Each event's actual type-safe payload is exactly one of these, selected via
`std::variant` on the unifying node:

```cpp
// torch/csrc/profiler/collection.h:377-436 (selected members)
struct TORCH_API Result : public std::enable_shared_from_this<Result> {
  template <typename... Args>
  [[nodiscard]] static std::shared_ptr<Result> create(Args... args) {
    return std::shared_ptr<Result>(new Result(std::forward<Args>(args)...));
  }

  EventType tag() const {
    return visit([](const auto& i) { return deduceTag(i); });
  }

  std::string name() const;
  std::string overload_name() const;
  libkineto::ActivityType kinetoType() const;
  uint64_t correlationID() const;
  int64_t endTimeNS() const;
  uint64_t endTID() const;
  c10::DeviceType deviceType() const;

  int64_t start_time_ns_;
  uint64_t start_tid_;
  kineto::DeviceAndResource kineto_info_;
  std::variant<
      ExtraFields<EventType::TorchOp>,
      ExtraFields<EventType::Backend>,
      ExtraFields<EventType::Vulkan>,
      ExtraFields<EventType::Allocation>,
      ExtraFields<EventType::OutOfMemory>,
      ExtraFields<EventType::PyCall>,
      ExtraFields<EventType::PyCCall>,
      ExtraFields<EventType::Kineto>,
      ExtraFields<EventType::PythonGC>>
      extra_fields_;

  std::weak_ptr<Result> parent_;
  std::vector<std::shared_ptr<Result>> children_;
  bool finished_{false};
  bool hidden_{false};
  const torch::profiler::impl::kineto::activity_t* kineto_activity_{nullptr};
};
```

`parent_`/`children_` make this a **tree**, not a flat list — nested op
calls (e.g. `aten::linear` calling `aten::addmm`) are represented by actual
parent/child `Result` pointers, which is how TensorBoard's flame-graph-style
call tree and the `torch_tb_profiler` "Operator" view (`../README.md`
Finding 4) get built without re-parsing timestamps.

## 2. Where events live before export (per-thread, columnar, not objects yet)

Allocating a full `Result` object (with its `variant`, `weak_ptr`, etc.) on
every single op call would be far too slow for the hot path. Instead, each
thread keeps one `ThreadLocalSubqueue` with **separate, flat, append-only
lists per `EventType`**:

```cpp
// torch/csrc/profiler/collection.h:589-654 (abridged)
class TORCH_API ThreadLocalSubqueue {
  ...
  struct TorchOpStorage {
    // NB: This is a destructive operation.
    void materialize(
        std::vector<std::shared_ptr<Result>>& out,
        std::vector<ProfilerStepInfo>& step_info,
        const std::function<c10::time_t(c10::approx_time_t)>& time_converter,
        const uint64_t tid,
        const kineto::DeviceAndResource& kineto_info);

    using event_t = KinetoObserverContext::Event;
    class OpList : public AppendOnlyList<event_t, BlockSize, EventBlock> {
      ...
    } op_events_;

    InputOutputEncoder inputs_outputs_;             // report_input_shapes
    AppendOnlyList<jit_stack_t, BlockSize> jit_stack_;     // with_stack (JIT)
    AppendOnlyList<jit_modules_t, BlockSize> jit_modules_; // with_modules
    AppendOnlyList<extra_args_t, BlockSize> extra_args_;   // with_flops
    AppendOnlyList<extra_meta_t, BlockSize> extra_meta_;
    AppendOnlyList<kwinputs_t, BlockSize> kwinputs_;
    AppendOnlyList<FallbackPair, BlockSize> device_fallback_;
  } torch_ops_;

  AppendOnlyList<ExtraFields<EventType::Backend>, BlockSize> backend_events_;
  AppendOnlyList<ExtraFields<EventType::Vulkan>::raw_event_t, BlockSize> vulkan_events_;
  AppendOnlyList<RawAllocation, BlockSize> allocations_;             // reportMemoryUsage
  AppendOnlyList<ExtraFields<EventType::OutOfMemory>, BlockSize> ooms_; // reportOOMs
  AppendOnlyList<std::pair<python_tracer::TraceKey, c10::approx_time_t>, BlockSize> py_calls_;
  AppendOnlyList<std::pair<std::string, c10::approx_time_t>, BlockSize> pythongc_;
};
```
(`collection.h:532-661` for the full class; `BlockSize = 512`,
`collection.h:588`.)

Key points:

- **One queue per OS thread** (`ThreadLocalSubqueue`), not one global queue
  — avoids cross-thread locking on the hot path. `RecordQueue::getSubqueue()`
  (`collection.h:669`) looks up or creates the calling thread's queue.
- **`AppendOnlyList`** is a chunked array (`BlockSize=512` elements per
  chunk) — an append is just a pointer bump, no heap churn per event,
  no vtable, no `shared_ptr` yet.
- **Memory events use the exact same mechanism** as op events —
  `allocations_`/`ooms_` are `AppendOnlyList<RawAllocation>` /
  `AppendOnlyList<ExtraFields<OutOfMemory>>`, fed from the allocator's
  `reportMemoryUsage`/`reportOOMs` hooks (comments at
  `collection.h:647,650`), not from RecordFunction at all — they're a
  separate instrumentation point (the c10 allocator), unified into the same
  `EventType`/`Result` scheme only at `materialize()` time.
- Nothing here is a `Result` yet. `TorchOpStorage::materialize()`
  (`collection.h:592-597`, called from `RecordQueue::getRecords()`,
  `collection.h:674-680`) is the one place that walks all these flat lists
  and actually constructs `std::shared_ptr<Result>` nodes, wires up
  parent/child links, and attaches the correlation ids that the Kineto
  device-activity trace will later be matched against.

## 3. Does the event format change per activity? Short answer: no (for what you asked), with one caveat

**CPU-observed events (everything in §1/§2): one format, always.** Whether
you request `activities=[CPU]`, `[CPU, XPU]`, or (unverified)
`[CPU, CUDA]`, every ATen op still goes through the identical
`RecordFunction` → `ThreadLocalSubqueue::begin_op` → `Result` pipeline.
Requesting `XPU` or `CUDA` does not change how CPU ops are recorded — it
only turns on an *additional*, independent device-activity collector (next
paragraph). This is directly visible in the two traces already captured in
`../traces/`: the `cpu_op`/`python_function` event shapes in
`traces/cpu/*.pt.trace.json` and `traces/xpu/*.pt.trace.json` are
structurally identical; the XPU trace simply has *more* categories, not a
different shape for the categories CPU also has (see
`docs/02-cpu-path.md`'s field-by-field comparison).

**Device-side events: one *stored* format, but genuinely different *wire*
formats underneath — the convergence point is `GenericTraceActivity`.**
This is the caveat. Each vendor plugin speaks its own protocol to its own
driver:

- XPU: PTI delivers `pti_view_record_kernel` / `_memory_copy` / `_memory_fill`
  / `_api` / `_overhead` (`pti_view.h:170-351`, see `docs/03-xpu-path.md`).
- CUDA (`[unverified]`): CUPTI delivers `CUpti_Activity`-derived structs
  (`CUpti_ActivityKernel*`, `CUpti_ActivityMemcpy*`, etc. — see
  `docs/04-cuda-path.md`).

These wire structs are **not** the same shape (different field names,
different vendor conventions) — but each plugin's `handle*Activity()`
method translates its vendor struct into the one shared
`libkineto::GenericTraceActivity`:

```cpp
// include/kineto/GenericTraceActivity.h:135-152
int64_t startTime{0};
int64_t endTime{0};
int32_t id{0};
int32_t device{0};
int32_t resource{0};
int32_t threadId{0};
ActivityType activityType;
std::string activityName;
struct Flow {
  Flow() : id(0), type(0), start(0) {}
  uint32_t id;
  uint32_t type : 4;
  uint32_t start : 1;
} flow;
const ITraceActivity* linked{nullptr};
```

And the `ActivityType` tag on that struct (`libkineto::ActivityType`, not
to be confused with `torch::profiler::impl::ActivityType` from
`observer.h:14-21` — two distinct enums with the same name in different
namespaces) is itself vendor-agnostic where it can be:

```cpp
// include/kineto/ActivityType.h:20-47 (abridged)
enum class ActivityType {
  CPU_OP = 0,
  USER_ANNOTATION,
  GPU_USER_ANNOTATION,
  GPU_MEMCPY,
  GPU_MEMSET,
  CONCURRENT_KERNEL,     // <- on-device kernels: BOTH XPU and CUDA kernels
                          //    land here, there is no separate XPU_KERNEL
  EXTERNAL_CORRELATION,
  CUDA_RUNTIME,
  CUDA_DRIVER,
  CPU_INSTANT_EVENT,
  PYTHON_FUNCTION,
  OVERHEAD,
  ...
  XPU_RUNTIME,           // host-side XPU runtime events (vendor-specific)
  ...
};
```

So: `CONCURRENT_KERNEL` is shared by XPU and CUDA kernels alike (confirmed
— there is no `XPU_KERNEL`/`CUDA_KERNEL` split in this enum); only the
*runtime*-call-level categories (`XPU_RUNTIME` vs `CUDA_RUNTIME`/
`CUDA_DRIVER`) distinguish vendors, because host-side API conventions
genuinely differ. **Net answer:** the event *taxonomy* and *stored struct*
are shared across all three activities; only the raw wire protocol each
vendor plugin consumes before translating into that shared struct differs.

## 4. Mapping back to the exported JSON categories

`../README.md`'s Finding 3 lists the exported category breakdown for the
original XPU trace (`cpu_op`, `python_function`, `ac2g`, `fwdbwd`,
`xpu_runtime`, `kernel`, `Trace`). Now that the new memory-profiling-enabled
traces exist (`kineto-deep-dive/traces/`), there's one more category,
`cpu_instant_event`, confirmed present:

```
$ python3 -c "
import json
d = json.load(open('traces/cpu/x4020c5s4b0n0_731947.1790952803600255990.pt.trace.json'))
cats = {}
for e in d['traceEvents']:
    cats[e.get('cat','?')] = cats.get(e.get('cat','?'), 0) + 1
print(cats)
"
{'cpu_op': 54, 'fwdbwd': 8, 'cpu_instant_event': 10, 'python_function': 216, '?': 60, 'Trace': 2}
```

Category → source mapping, now complete:

| JSON `cat` | `EventType` / origin | where it comes from |
|---|---|---|
| `cpu_op` | `EventType::TorchOp` | `ThreadLocalSubqueue::begin_op`, §2 above |
| `python_function` | `EventType::PyCall`/`PyCCall` | Python call-stack tracer (`with_stack=True`) |
| `cpu_instant_event` | `EventType::Allocation` (`name: "[memory]"`) | allocator `reportMemoryUsage`, §2 above — see `docs/02-cpu-path.md` for the exact event |
| `fwdbwd` / `ac2g` | `ExtraFields<Kineto>::Flow` (`collection.h:361-365`) | forward/backward correlation arrows, built during `materialize()` |
| `xpu_runtime` | `libkineto::ActivityType::XPU_RUNTIME` | PTI `pti_view_record_api`, via `XpuptiActivityProfilerSession::handleRuntimeActivity` |
| `kernel` | `libkineto::ActivityType::CONCURRENT_KERNEL` | PTI `pti_view_record_kernel`, via `handleKernelActivity` — see `docs/03-xpu-path.md` |
| `Trace` | trace-level metadata, not a `Result`/activity at all | written once by the exporter, not per-event |

## What's next

- `docs/02-cpu-path.md` — the real `[memory]`/`cpu_instant_event` entry from
  the trace above, plus a real `aten::addmm` op, each annotated
  field-by-field against the structs in §1/§2.
- `docs/03-xpu-path.md` — a real `kernel`-category event, annotated against
  `pti_view_record_kernel`.
- `docs/04-cuda-path.md` — the CUDA analogue of §3's wire-format claim, from
  upstream source only.
