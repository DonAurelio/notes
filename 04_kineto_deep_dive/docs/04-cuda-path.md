# CUDA path: the same walkthrough as XPU, from upstream source only

**STATUS: UNVERIFIED — no NVIDIA/CUDA hardware exists on this system.**
Everything in this document is derived from reading `pytorch/kineto`'s
upstream source on GitHub, at the exact commit this PyTorch build pins —
**no code here has been compiled or executed**. Treat every claim as a
prediction to be checked against a real trace. `example/profile_cuda.py`
(already written, structurally identical to `example/profile_xpu.py`) is
ready to run on a CUDA-capable system to do that check — see §6.

**PyTorch version:** `2.10.0a0+git449b176`, full commit
`449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`. The `third_party/kineto`
submodule pointer at that exact PyTorch commit (resolved via GitHub's
Contents API: `GET /repos/pytorch/pytorch/contents/third_party/kineto?ref=449b1768...`)
is:

```
pytorch/kineto @ 31f85df8fbd89c188f14ef10f1ec65379786b943
```

All `libkineto/src/*` citations below are `path:line` against that exact
commit — fetched directly from
`https://raw.githubusercontent.com/pytorch/kineto/31f85df8fbd89c188f14ef10f1ec65379786b943/libkineto/src/<file>`,
not from memory of "typical" CUPTI code. `CUpti_Activity*` struct field
names (e.g. `kernel->name`, `kernel->correlationId`) are cited only where
kineto's own source visibly accesses them — the full struct definitions
live in NVIDIA's proprietary `cupti_activity.h` (shipped with the CUDA
Toolkit), which is not in this repository and not available to check on
this system, so no NVIDIA-header line numbers are cited anywhere in this
document.

## 1. The structural parallel to docs/03-xpu-path.md

Every concept in the XPU doc has a direct CUDA-side counterpart, same
layering, different plugin:

| Concept | XPU (`docs/03-xpu-path.md`) | CUDA (this doc) |
|---|---|---|
| Kineto plugin class | `libkineto::XpuptiActivityApi` | `libkineto::CuptiActivityApi` (`libkineto/src/CuptiActivityApi.h:38-119`) |
| Vendor profiling library | Intel PTI (`libpti_view.so`) | NVIDIA CUPTI (`libcupti.so`, part of the CUDA Toolkit) |
| Enable call | `ptiViewEnable(PTI_VIEW_DEVICE_GPU_KERNEL)` | `cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)` |
| Buffer callback registration | `ptiViewSetCallbacks(...)` | `cuptiActivityRegisterCallbacks(bufferRequestedTrampoline, bufferCompletedTrampoline)` |
| Pull next record | `ptiViewGetNextRecord(...)` | `cuptiActivityGetNextRecord(...)` |
| Kernel record struct | `pti_view_record_kernel` (`pti_view.h:170-198`) | `CUpti_ActivityKernel4` (NVIDIA CUDA Toolkit header, not in this repo) |
| Translation function | `XpuptiActivityProfilerSession::handleKernelActivity(pti_view_record_kernel const*, ...)` | `CuptiActivityProfiler::handleCuptiActivity(const CUpti_Activity*, ActivityLogger*)` dispatching to `handleGpuActivity<CUpti_ActivityKernel4>(...)` |
| Driver hooked | Level Zero (`zeInit`) | CUDA driver API |
| `libkineto::ActivityType` tag | `CONCURRENT_KERNEL` (shared, see `docs/01-event-model.md` §3) | `CONCURRENT_KERNEL` (same enum value — confirmed below) |

## 2. Activation: `CuptiActivityApi::enableCuptiActivities`

```cpp
// libkineto/src/CuptiActivityApi.h:38-70 (commit 31f85df8)
class CuptiActivityApi {
 public:
  enum CorrelationFlowType { Default, User };
  ...
  CuptiActivityApi() = default;
  virtual ~CuptiActivityApi() = default;

  static CuptiActivityApi& singleton();

  static void pushCorrelationID(int id, CorrelationFlowType type);
  static void popCorrelationID(CorrelationFlowType type);

  void enableCuptiActivities(
      const std::set<ActivityType>& selected_activities,
      bool enablePerThreadBuffers = false);
  void disableCuptiActivities(
      const std::set<ActivityType>& selected_activities);
  void clearActivities();
  void flushActivities();
  void teardownContext();

  virtual std::unique_ptr<CuptiActivityBufferMap> activityBuffers();

  virtual const std::pair<int, size_t> processActivities(
      CuptiActivityBufferMap&,
      const std::function<void(const CUpti_Activity*)>& handler);
  ...
};
```

This is the exact structural analogue of `XpuptiActivityApi` from
`docs/00-overview.md` §1 — same singleton pattern, same
`enable*Activities(std::set<ActivityType>&)` signature shape.

The real activation sequence, from `libkineto/src/CuptiActivityApi.cpp`
(commit `31f85df8`), inside `enableCuptiActivities` — this is the function
Kineto calls when `profile(activities=[CPU, CUDA])` is requested:

```cpp
// libkineto/src/CuptiActivityApi.cpp:349-387 (abridged)
CUPTI_CALL(cuptiActivityRegisterCallbacks(
    bufferRequestedTrampoline, bufferCompletedTrampoline));

externalCorrelationEnabled_ = false;
for (const auto& activity : selected_activities) {
  if (activity == ActivityType::GPU_MEMCPY) {
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  }
  if (activity == ActivityType::GPU_MEMSET) {
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
  }
  if (activity == ActivityType::CONCURRENT_KERNEL) {
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  }
  if (activity == ActivityType::EXTERNAL_CORRELATION) {
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
    externalCorrelationEnabled_ = true;
  }
  if (activity == ActivityType::CUDA_RUNTIME) {
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  }
  ...
}
```

This maps `libkineto::ActivityType` values (the same shared enum discussed
in `docs/01-event-model.md` §3 — `CONCURRENT_KERNEL`, `CUDA_RUNTIME`,
`EXTERNAL_CORRELATION`, etc., `ActivityType.h:20-47`) one-to-one onto
`cuptiActivityEnable(CUPTI_ACTIVITY_KIND_*)` calls — exactly analogous to
how `XpuptiActivityApi::enableXpuptiActivities` maps the same enum onto
`ptiViewEnable(PTI_VIEW_*)` calls (`docs/03-xpu-path.md` §1). The buffer
callback registration (`cuptiActivityRegisterCallbacks`, line 349-350) is
the direct counterpart of PTI's `ptiViewSetCallbacks`
(`pti_view.h:400-402`) — same push-based buffering model: CUPTI fills a
caller-provided buffer and invokes `bufferCompletedTrampoline` when full.

Pulling records back out works the same way as PTI's `ptiViewGetNextRecord`:

```cpp
// libkineto/src/CuptiActivityApi.cpp:102-109 (abridged)
static bool nextActivityRecord(
    uint8_t* buffer,
    size_t valid_size,
    CUpti_Activity*& record) {
  CUptiResult status = CUPTI_CALL_NOWARN(
      cuptiActivityGetNextRecord(buffer, valid_size, &record));
  ...
}
```

## 3. The kernel record and its translation into `GenericTraceActivity`

Where XPU's `handleKernelActivity(pti_view_record_kernel const*, ...)`
reads a PTI struct directly, CUDA dispatches through one generic entry
point first — `handleCuptiActivity`, declared in
`libkineto/src/CuptiActivityProfiler.h:387-409` (header excerpt):

```cpp
// libkineto/src/CuptiActivityProfiler.h:387-409 (commit 31f85df8)
#ifdef HAS_CUPTI
  // Process generic CUPTI activity
  void handleCuptiActivity(
      const CUpti_Activity* record,
      ActivityLogger* logger);
  // Process specific GPU activity types
  void handleCorrelationActivity(
      const CUpti_ActivityExternalCorrelation* correlation);
  void handleRuntimeActivity(
      const CUpti_ActivityAPI* activity,
      ActivityLogger* logger);
  void handleDriverActivity(
      const CUpti_ActivityAPI* activity,
      ActivityLogger* logger);
  void handleOverheadActivity(
      const CUpti_ActivityOverhead* activity,
      ActivityLogger* logger);
  void handleCudaEventActivity(
      const CUpti_ActivityCudaEventType* activity,
      ActivityLogger* logger);
  void handleCudaSyncActivity(
      const CUpti_ActivitySynchronization* activity,
      ActivityLogger* logger);
  template <class T>
  void handleGpuActivity(const T* act, ActivityLogger* logger);
#endif
```

...and its implementation (`libkineto/src/CuptiActivityProfiler.cpp`, the
`handleCuptiActivity` definition starting at line 904) is a `switch` on
`record->kind` that dispatches each CUPTI activity kind to its own
handler — the kernel case:

```cpp
// libkineto/src/CuptiActivityProfiler.cpp:904-926 (commit 31f85df8)
void CuptiActivityProfiler::handleCuptiActivity(
    const CUpti_Activity* record,
    ActivityLogger* logger) {
  switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION:
      handleCorrelationActivity(
          reinterpret_cast<const CUpti_ActivityExternalCorrelation*>(record));
      break;
    case CUPTI_ACTIVITY_KIND_RUNTIME:
      handleRuntimeActivity(
          reinterpret_cast<const CUpti_ActivityAPI*>(record), logger);
      break;
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
      auto kernel = reinterpret_cast<const CUpti_ActivityKernel4*>(record);
      // Register all kernels launches so we could correlate them with other
      // events.
      KernelRegistry::singleton()->recordKernel(
          kernel->deviceId, demangle(kernel->name), kernel->correlationId);
      handleGpuActivity(kernel, logger);
      updateCtxToDeviceId(kernel);
      break;
    }
    case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION:
      handleCudaSyncActivity(
          reinterpret_cast<const CUpti_ActivitySynchronization*>(record),
          logger);
      break;
    ...
```

Three fields are visibly read directly off `CUpti_ActivityKernel4` here —
`kernel->deviceId`, `kernel->name`, `kernel->correlationId` — confirming
(without needing NVIDIA's own header) that the CUDA kernel record carries
at minimum a device id, a (mangled) kernel name, and a correlation id, the
same three pieces of information `pti_view_record_kernel` exposes as
`_device_uuid`/`_pci_address`, `_name`, and `_correlation_id`
(`docs/03-xpu-path.md` §2). The templated `handleGpuActivity<T>(act, logger)`
call is the function that actually builds the `GenericTraceActivity` —
structurally the same role as XPU's `handleKernelActivity`, just reached
through one extra dispatch layer because CUPTI multiplexes many activity
kinds through a single `CUpti_Activity*` base pointer (`record->kind`)
rather than PTI's separately-typed buffers.

## 4. Predicted exported JSON shape

Based on §1-3 and the shared `libkineto::ActivityType`/`GenericTraceActivity`
convergence point established in `docs/01-event-model.md` §3, a CUDA trace
from `example/profile_cuda.py` should show:

- `cpu_op` / `python_function` / `cpu_instant_event` (`[memory]`) —
  **identical in shape** to the CPU-only and XPU traces already captured
  (`docs/02-cpu-path.md`), since these come entirely from PyTorch's own
  `RecordFunction`/`ThreadLocalSubqueue` machinery and do not depend on
  which device backend is active.
- `kernel` — CUDA kernel executions, same `libkineto::ActivityType::CONCURRENT_KERNEL`
  tag as the XPU `gemm_kernel` events, but the kernel name will reflect
  whatever CUDA/cuBLAS kernel implements `addmm`/`mm` on this hardware
  (likely something like `ampere_sgemm_*`/`volta_sgemm_*` or a cuBLAS
  internal kernel name, not `gemm_kernel` — XPU's SYCL kernel naming is
  oneDNN/oneMKL-specific and does not carry over).
- `cuda_runtime` (not `xpu_runtime`) — the host-side CUDA runtime API calls
  (`cudaLaunchKernel`, `cudaMemcpyAsync`, etc. — the `CUPTI_ACTIVITY_KIND_RUNTIME`
  activities enabled at `CuptiActivityApi.cpp:374`), analogous to XPU's
  `urEnqueueKernelLaunch`.
- `ac2g`/`fwdbwd` — should appear identically, since these come from
  `ExtraFields<EventType::Kineto>::Flow` (`collection.h:361-365`), which is
  vendor-agnostic.
- Correlation chain: expect the same 3-hop pattern documented in
  `docs/03-xpu-path.md` §4 (`cpu_op` → `cuda_runtime` → `kernel`, linked by
  a shared `correlation`/`External id` pair), since that mechanism lives in
  PyTorch's `ExtraFields<EventType::Kineto>` / Kineto's
  `EXTERNAL_CORRELATION` activity type, not in the vendor plugin.

## 5. Why this can't be checked here

`torch.cuda.is_available()` is unconditionally `False` on this build —
confirmed by inspecting the installed `torch` package on this Aurora node,
which only ships `libtorch_xpu.so`/`libtorch_cpu.so`, no `libtorch_cuda.so`,
and has no CUDA Toolkit (`libcupti.so`) installed anywhere on the system.
There is no equivalent of the XPU path's live `zeInit()` failure to even
reproduce here — the CUDA activity plugin code is simply not compiled into
this `libtorch_cpu.so` at all (unlike XPU's `XpuptiActivityApi`, which
*is* compiled in but fails at runtime for lack of hardware).

## 6. Verification checklist (for a CUDA-capable system)

1. `source env.sh` (or the target system's equivalent module/venv setup —
   this project's `env.sh` is Aurora/oneAPI-specific and won't apply
   elsewhere).
2. Confirm `python3 -c "import torch; print(torch.cuda.is_available())"`
   prints `True`.
3. `cd example && python3 profile_cuda.py` — this is already written and
   untouched from the structure validated in `docs/00-overview.md`'s
   planning (same shape as `profile_xpu.py`, `torch.cuda` substituted for
   `torch.xpu`).
4. Copy the resulting `traces/cuda/*.pt.trace.json` back into this
   project's `traces/cuda/` (see that directory's `README.md` stub).
5. Repeat the category-breakdown check from `docs/01-event-model.md` §4
   (`python3 -c "import json; ... cats = ..."`) and compare against §4's
   predictions above — specifically: is the kernel category tagged
   `kernel` or something else, is the runtime category named
   `cuda_runtime`, and does the correlation id actually link all three
   hops the way `docs/03-xpu-path.md` §4 demonstrates for XPU.
6. If anything differs from §1-4's predictions, update this document with
   the corrected, evidence-based finding — don't leave the prediction
   standing once a real trace contradicts it.
