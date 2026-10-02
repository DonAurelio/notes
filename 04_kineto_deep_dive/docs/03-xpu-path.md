# XPU path: a real kernel, traced from the Level Zero driver up to the stored event

**PyTorch version:** `2.10.0a0+git449b176` (commit
`449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`). Citations are `path:line` into:

```
$TORCH_DIR = /opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/lib/python3.12/site-packages/torch
PTI_DIR    = /opt/aurora/26.26.0/spack/unified/1.1.1/install/linux-x86_64/pti-gpu-0.17.0-hi6zrvx
             (confirmed as the install actually linked: `ldd libtorch_cpu.so | grep pti`)
```

**Data source:** `../traces/xpu/x4020c5s4b0n0_732211.1790952812259477428.pt.trace.json`,
recorded on Aurora compute node `x4020c5s4b0n0` by running
`kineto-deep-dive/example/profile_xpu.py`. This is the most novel part of
the deep dive: unlike `docs/02-cpu-path.md` (all PyTorch-internal), this
path crosses into a separate system library (PTI) talking to a hardware
driver (Level Zero) neither PyTorch nor Kineto's headers describe.

## 1. Activation: turning on XPU kernel collection

When `profile(activities=[CPU, XPU])` is requested, Kineto's XPU plugin —
`libkineto::XpuptiActivityApi`, compiled into `libtorch_cpu.so` (confirmed
via `nm -D libtorch_cpu.so | c++filt`, see `docs/00-overview.md` §1) — calls
into PTI's "view" API to turn on kernel collection:

```c
// PTI_DIR/include/pti/pti_view.h:32-41 (pti_view_kind enum, abridged)
typedef enum _pti_view_kind {
  PTI_VIEW_INVALID = 0,
  PTI_VIEW_DEVICE_GPU_KERNEL = 1,       // <- what we enable below
  PTI_VIEW_DEVICE_CPU_KERNEL = 2,
  PTI_VIEW_DRIVER_API = 3,
  PTI_VIEW_RUNTIME_API = 6,             // <- "xpu_runtime" category, §4
  PTI_VIEW_EXTERNAL_CORRELATION = 7,
  PTI_VIEW_DEVICE_GPU_MEM_COPY = 8,
  PTI_VIEW_DEVICE_GPU_MEM_FILL = 9,
} pti_view_kind;
```

```c
// PTI_DIR/include/pti/pti_view.h:400-402, :410, :444-445
pti_result PTI_EXPORT ptiViewSetCallbacks(
    pti_fptr_buffer_requested fptr_bufferRequested,
    pti_fptr_buffer_completed fptr_bufferCompleted);

pti_result PTI_EXPORT ptiViewEnable(pti_view_kind view_kind);

pti_result PTI_EXPORT ptiViewGetNextRecord(
    uint8_t* buffer, size_t valid_bytes, pti_view_record_base** record);
```

`XpuptiActivityApi::enableXpuptiActivities` (symbol confirmed in
`libtorch_cpu.so`) calls `ptiViewSetCallbacks` once (registering a
buffer-allocation callback and a buffer-ready callback — PTI's buffering is
push-based: it fills caller-provided buffers and calls back when one is
full), then `ptiViewEnable(PTI_VIEW_DEVICE_GPU_KERNEL)` (and
`PTI_VIEW_RUNTIME_API`, `PTI_VIEW_EXTERNAL_CORRELATION`, etc., one call per
activity type requested). From here PTI hooks the Level Zero driver
directly — this is the **same `zeInit()` call path** documented in
`../README.md` Finding 0 as the reason the login node crashes
(`PTI_ERROR_INTERNAL`/`Unable to initialize Level Zero driver(s)` — no
Level Zero devices to hook on a node with zero XPUs).

## 2. The record PTI hands back: `pti_view_record_kernel`

Once the kernel this example triggers (`gemm_kernel`, from `aten::mm`/
`aten::addmm` — see `docs/02-cpu-path.md` §1) actually runs on the GPU, PTI
buffers a `pti_view_record_kernel` and Kineto drains it with
`ptiViewGetNextRecord`:

```c
// PTI_DIR/include/pti/pti_view.h:170-198
typedef struct pti_view_record_kernel {
  pti_view_record_base _view_kind;                //!< Base record
  pti_backend_queue_t _queue_handle;               //!< Device back-end queue handle
  pti_backend_ctx_t _context_handle;               //!< Context handle
  const char* _name;                               //!< Kernel name
  const char* _source_file_name;
  uint64_t _source_line_number;
  uint64_t _kernel_id;                             //!< Kernel instance ID
  uint32_t _correlation_id;                        //!< ID correlating this record with other Views
  uint32_t _thread_id;
  char _pci_address[PTI_MAX_PCI_ADDRESS_SIZE];
  uint8_t _device_uuid[PTI_MAX_DEVICE_UUID_SIZE];
  uint64_t _append_timestamp;                      //!< kernel appended to cmd list, ns
  uint64_t _start_timestamp;                       //!< kernel start on device, ns
  uint64_t _end_timestamp;                         //!< kernel completion on device, ns
  uint64_t _submit_timestamp;                      //!< kernel cmd list submission, ns
  uint64_t _sycl_task_begin_timestamp;
  uint64_t _sycl_enqk_begin_timestamp;
  uint64_t _sycl_node_id;
  uint64_t _sycl_queue_id;                         //!< Device front-end queue id
  uint32_t _sycl_invocation_id;
} pti_view_record_kernel;
```

`XpuptiActivityProfilerSession::handleKernelActivity(pti_view_record_kernel const*, ActivityLogger*)`
(symbol confirmed via `nm -D libtorch_cpu.so | c++filt`) is the function
that reads this struct and constructs a `libkineto::GenericTraceActivity`
from it (`GenericTraceActivity.h:135-152`, cited in `docs/00-overview.md`
§1 and `docs/01-event-model.md` §3).

## 3. The actual recorded event, annotated field-by-field

Two `gemm_kernel` events exist in the trace (the forward matmul and the
backward-pass matmul). The first one:

```json
{
  "ph": "X",
  "cat": "kernel",
  "name": "gemm_kernel",
  "pid": 0,
  "tid": 0,
  "ts": 95786114294.336,
  "dur": 8.96,
  "args": {
    "External id": 5,
    "kernel_id": 1,
    "l0 queue": "0x000055d08b8fc138",
    "sycl queue": 64,
    "context": "0x000055d08918fc78",
    "correlation": 4131,
    "device": 0,
    "submitted": 1790952812114058336,
    "appended": 1790952812114058336
  }
}
```

| JSON field | Comes from | Citation |
|---|---|---|
| `"name": "gemm_kernel"` | `pti_view_record_kernel::_name` | `pti_view.h:174` |
| `args."kernel_id"` | `pti_view_record_kernel::_kernel_id` | `pti_view.h:178` |
| `args."correlation"` | `pti_view_record_kernel::_correlation_id` | `pti_view.h:180` |
| `args."l0 queue"` | `pti_view_record_kernel::_queue_handle` (Level Zero queue handle, printed as a pointer) | `pti_view.h:172` |
| `args."context"` | `pti_view_record_kernel::_context_handle` | `pti_view.h:173` |
| `args."sycl queue"` | `pti_view_record_kernel::_sycl_queue_id` | `pti_view.h:196` |
| `args."device"` | derived from the device/PCI info (`_pci_address`/`_device_uuid`) resolved to a logical index by the Kineto session, not a 1:1 struct field | `pti_view.h:183-184` |
| `args."appended"` | `pti_view_record_kernel::_append_timestamp` | `pti_view.h:186` |
| `args."submitted"` | `pti_view_record_kernel::_submit_timestamp` | `pti_view.h:190` |
| `"ts"` / `"dur"` | derived from `_start_timestamp`/`_end_timestamp` | `pti_view.h:188-189` |
| `args."External id": 5` | **not** a PTI field — this is PyTorch's own correlation id, attached by Kineto when it links this device activity back into the `Result` tree (see §4) | n/a (Kineto/PyTorch-side, not PTI) |

## 4. The full correlation chain: CPU op → runtime call → device kernel

This is the concrete evidence for the "two parallel structures stitched by
correlation id" claim in `docs/00-overview.md` §4 and
`docs/01-event-model.md` §4. Three events from the *same* trace, same
`correlation`/`External id` values, in causal order:

**1. The CPU-side op** (`aten::addmm`, recorded by RecordFunction exactly as
in `docs/02-cpu-path.md`):

```json
{
  "cat": "cpu_op", "name": "aten::addmm",
  "ts": 95786069998.433, "dur": 44332.734,
  "args": {"External id": 5, "Sequence number": 1, ...}
}
```

**2. The host-side runtime call** (PTI's `PTI_VIEW_RUNTIME_API` view,
`pti_view.h:39`, struct `pti_view_record_api` at `pti_view.h:341-351` —
`_api_group`/`_api_id` identify the exact SYCL/Level-Zero entry point,
`_correlation_id` at `pti_view.h:349` is the same id the kernel record
carries):

```json
{
  "cat": "xpu_runtime", "name": "urEnqueueKernelLaunch",
  "ts": 95786113648.115, "dur": 640.29,
  "args": {"External id": 5, "correlation": 4131}
}
```

**3. The device kernel** (§3 above, repeated for alignment):

```json
{
  "cat": "kernel", "name": "gemm_kernel",
  "ts": 95786114294.336, "dur": 8.96,
  "args": {"External id": 5, "correlation": 4131, ...}
}
```

`correlation: 4131` is identical across the runtime-API record and the
kernel record (both are PTI-side ids, `_correlation_id` on their respective
structs); `"External id": 5` is identical across *all three*, including the
original CPU op — this is the field Kineto/PyTorch itself assigns (not a
PTI field) to thread the device activity back into the `Result` tree's
`ExtraFields<EventType::Kineto>::correlation_id_`
(`collection.h:370`, cited in `docs/00-overview.md` §4). An `ac2g`
flow-start event ties the same id into the visual flow arrow TensorBoard
draws between the CPU op and its device activity:

```json
{"ph": "s", "id": 4131, "ts": 95786113648.115, "cat": "ac2g", "name": "ac2g"}
```

So the causal order, with real timestamps from this trace (chrome-trace
JSON's `ts`/`dur` fields are **microseconds**, not nanoseconds):
`aten::addmm` starts at `...069998.433` (CPU dispatch) →
`urEnqueueKernelLaunch` at `...113648.115` (host enqueues the kernel onto
the Level Zero queue, **~43.65ms** later — this gap is dominated by
`aten::addmm`'s own CPU-side work, matching its own recorded `dur: 44332.734us`
≈ 44.3ms, before it reaches the XPU backend, not wait time on the device) →
`gemm_kernel` actually executes at `...114294.336` (**~646us** after
enqueue — the real device-side launch latency) and finishes `8.96us` later.

## 5. Why none of this exists on the login node

Already established with direct evidence in `../README.md` Finding 0:
`ptiViewEnable`/PTI's Level Zero backend init calls `zeInit()`, which
fails with no XPU devices present:

```
PTI:[...] ze_driver_init.cc:92 zeInit returned: 2013265921.
PTI:[...] ze_collector.h:192 Unable to initialize Level Zero driver(s)
```

This doc adds the missing piece: that failure happens *inside the very
first PTI call this section describes* (`ptiViewEnable`, §1) — which is why
even `activities=[CPU]`-only profiling fails there too (per
`docs/00-overview.md`/`../README.md`: Kineto's XPU activity collector is not
gated per-requested-activity, so `enableXpuptiActivities`/`ptiViewEnable`
still runs during `_prepare_profiler` regardless of which activities were
asked for).

## What's next

- `docs/04-cuda-path.md` — the same walkthrough for CUDA, from upstream
  `pytorch/kineto` source only (no NVIDIA hardware on this system) —
  `CuptiActivityApi` stands in for `XpuptiActivityApi`, `CUpti_Activity*`
  structs stand in for `pti_view_record_*`, and `cudaLaunchKernel`-family
  runtime calls stand in for `urEnqueueKernelLaunch`.
