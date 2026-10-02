# CPU path: a real op and a real memory event, traced end to end

**PyTorch version:** `2.10.0a0+git449b176` (commit
`449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`). Citations are `path:line` into:

```
$TORCH_DIR = /opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/lib/python3.12/site-packages/torch
```

**Data source:** `../traces/cpu/x4020c5s4b0n0_731947.1790952803600255990.pt.trace.json`,
recorded on Aurora compute node `x4020c5s4b0n0` by running
`kineto-deep-dive/example/profile_cpu.py` (see `docs/00-overview.md` for the
activation path, `docs/01-event-model.md` for the classification this
section annotates against).

## 1. A real `aten::addmm` op event

`MyModel.forward` (`example/model.py`) is `torch.relu(self.linear(x))`.
`nn.Linear.forward` calls `F.linear`, which dispatches to `aten::addmm`
(bias + matmul fused) — this is the single largest self-time entry in every
CPU table printed so far (`../README.md` Finding 1, and the regenerated run
in `docs/00-overview.md`'s context). Pulled directly out of the trace file:

```json
{
  "ph": "X",
  "cat": "cpu_op",
  "name": "aten::addmm",
  "pid": 731947,
  "tid": 731947,
  "ts": 95777561774.582,
  "dur": 12845.949,
  "args": {
    "External id": 5,
    "Sequence number": 1,
    "Fwd thread id": 0,
    "Record function id": 0,
    "Concrete Inputs": ["", "", "", "1", "1"],
    "Input type": ["float", "float", "float", "Scalar", "Scalar"],
    "Input Strides": [[1], [10, 1], [1, 10], [], []],
    "Input Dims": [[5], [1, 10], [10, 5], [], []],
    "Ev Idx": 4
  }
}
```

### Field-by-field: JSON → `Result`/`ExtraFields<TorchOp>`

| JSON field | Comes from | Citation |
|---|---|---|
| `"cat": "cpu_op"` | `Result::tag() == EventType::TorchOp` | `collection.h:30` (enum value), `collection.h:405-407` (`tag()`) |
| `"name": "aten::addmm"` | `TorchOpBasicFields::name_` | `collection.h:122` |
| `"ts"` (start, converted to wall-clock) | `Result::start_time_ns_` | `collection.h:417` |
| `"dur"` | `Result::endTimeNS() - start_time_ns_`, where `endTimeNS()` reads `ExtraFields<TorchOp>::end_time_ns_` | `collection.h:413` (accessor), `collection.h:170` (field) |
| `"tid"` | `TorchOpBasicFields::end_tid_` (or `start_tid_` on `Result`) | `collection.h:126`, `collection.h:418` |
| `args."Sequence number"` | `TorchOpBasicFields::sequence_number_` | `collection.h:117` |
| `args."Fwd thread id"` | `TorchOpBasicFields::forward_tid_` | `collection.h:118` |
| `args."Record function id"` | `TorchOpBasicFields::record_function_id_` | `collection.h:120` |
| `args."Concrete Inputs"` | `ExtraFields<TorchOp>::concrete_inputs_` (populated because `record_shapes=True` also turns on concrete-input capture for scalar args) | `collection.h:172` |
| `args."Input type"` / `"Input Dims"` / `"Input Strides"` | `ExtraFields<TorchOp>::inputs_` (a `std::vector<op_input_t>`, `op_input_t` = variant of `TensorMetadata`/`vector<TensorMetadata>`/`IValue`) | `collection.h:171` (field), `collection.h:104-108` (`op_input_t` variant) |
| `args."Ev Idx"` | debug index into the exporter's flat event list, not a `Result` field itself — added by the JSON-export step, not `collection.h` | n/a (export-time bookkeeping) |

Why these specific inputs: `aten::addmm(bias, mat1, mat2, beta=1, alpha=1)`
— 5 positional args. `Input Dims` confirms this directly against
`example/model.py`'s `nn.Linear(10, 5)` applied to a `(1, 10)` input:
`[5]` (bias), `[1, 10]` (x), `[10, 5]` (weight, pre-transposed by
`aten::t`/`aten::linear`), and the two `Scalar` args (`beta`, `alpha`,
`Concrete Inputs: "1", "1"`) have no shape (`[]`).

### Correlation to the forward/backward flow

The same op's correlation to autograd is visible as a separate `fwdbwd`
flow-start event with `id: 4` — matching `args."External id": 4` on a
different (`AddmmBackward0`) event elsewhere in the trace, not shown here
for brevity, but confirming the `fwdbwd`/`ac2g` mechanism from
`docs/01-event-model.md` §4 (`ExtraFields<Kineto>::Flow`,
`collection.h:361-365`):

```json
{"ph": "s", "id": 4, "pid": 731947, "tid": 731947, "ts": 95777561663.102, "cat": "fwdbwd", "name": "fwdbwd"}
```

## 2. A real memory `[memory]` event

Turning on `profile_memory=True` (the only change made to
`example/profile_cpu.py` relative to the parent `../example/profile_cpu.py`
— see `docs/00-overview.md` §2's `ProfilerConfig.profile_memory` field,
`observer.h:155`) produces `cpu_instant_event`/`[memory]` entries. Two
consecutive ones from the same tensor address, an allocate followed by a
deallocate:

```json
{
  "ph": "i", "cat": "cpu_instant_event", "s": "t", "name": "[memory]",
  "pid": 731947, "tid": 731947, "ts": 95777561818.085,
  "args": {
    "Total Reserved": 0, "Total Allocated": 20, "Bytes": 20,
    "Device Id": -1, "Device Type": 0, "Addr": 94119953593280,
    "finished": false, "Ev Idx": 54
  }
}
{
  "ph": "i", "cat": "cpu_instant_event", "s": "t", "name": "[memory]",
  "pid": 731947, "tid": 731947, "ts": 95777574749.462,
  "args": {
    "Total Reserved": 0, "Total Allocated": 20, "Bytes": -20,
    "Device Id": -1, "Device Type": 0, "Addr": 94119953593280,
    "finished": false, "Ev Idx": 56
  }
}
```

The same `Addr` (`94119953593280`) appears twice: `Bytes: 20` (allocate,
20 bytes — a `float32` bias tensor of shape `[5]`: `5 * 4 = 20` bytes,
matching the `addmm` bias input above) then `Bytes: -20` (deallocate) a few
microseconds later, once the op that used it finished.

### Field-by-field: JSON → `RawAllocation`/`ExtraFields<Allocation>`

| JSON field | Comes from | Citation |
|---|---|---|
| `"cat": "cpu_instant_event"`, `"name": "[memory]"` | `EventType::Allocation`, exporter's fixed label for this category (not a `collection.h` field — assigned during trace export) | `collection.h:33` (enum value) |
| `"ts"` | `RawAllocation::start_time_` | `collection.h:214` |
| `args."Bytes"` | `RawAllocation::alloc_size_` (signed: positive = allocate, negative = free) | `collection.h:216` |
| `args."Total Allocated"` | `RawAllocation::total_allocated_` | `collection.h:217` |
| `args."Total Reserved"` | `RawAllocation::total_reserved_` | `collection.h:218` |
| `args."Device Type"` | `RawAllocation::device_type_` (`0` = `c10::DeviceType::CPU`) | `collection.h:219`; enum value at `torch/headeronly/core/DeviceType.h:36` |
| `args."Device Id"` | `RawAllocation::device_index_` (`-1` here — no specific device index for a plain CPU allocation) | `collection.h:220` |
| `args."Addr"` | `RawAllocation::ptr_` (printed as an integer in the JSON export) | `collection.h:215` |

`ExtraFields<Allocation> : RawAllocation` (`collection.h:229-238`) adds two
more fields not shown in the exported JSON (`id_`, `allocation_id_` —
internal tensor/allocation identifiers used for memory-timeline
reconstruction, e.g. `export_memory_timeline()`, not surfaced in the
chrome-trace format itself).

This event stream comes from a **completely different instrumentation
point** than `aten::addmm` above: the c10 allocator's `reportMemoryUsage`
hook (`collection.h:647`, feeding `ThreadLocalSubqueue::allocations_`,
`collection.h:648`), not `RecordFunction`. The two only end up in the same
exported trace because both get folded into the same `Result` tree by
`TorchOpStorage::materialize()`/`RecordQueue::getRecords()`
(`docs/00-overview.md` §4) before export — confirming the "memory events use
the exact same storage mechanism but a different *source*" point made in
`docs/01-event-model.md` §2.

## 3. What's NOT in this CPU-only trace

Two categories that *do* appear in the XPU trace are confirmed absent here:
`kernel` and `ac2g`. This matches `docs/01-event-model.md`'s claim that CPU
events are unaffected by which device activities are requested — but it
also means a CPU-only run produces no device-activity trace at all (no
`ActivityTraceWrapper` content beyond CPU bookkeeping), since no vendor
plugin (`XpuptiActivityApi`/`CuptiActivityApi`) was ever enabled. See
`docs/03-xpu-path.md` for what changes when `ProfilerActivity.XPU` is added.

## What's next

- `docs/03-xpu-path.md` — the device-side analogue: a real `gemm_kernel`
  event from `traces/xpu/`, annotated against `pti_view_record_kernel`
  (`pti_view.h:170-198`), including the PTI activation/pull sequence that
  produces it.
- `docs/04-cuda-path.md` — the CUDA analogue, from upstream source only.
