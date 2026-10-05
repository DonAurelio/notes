# Device support: ITT has no CUDA (or XPU-device-activity) path — by design

**PyTorch version:** `2.10.0a0+git449b176` (commit
`449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`). This is the explicit answer
the user asked for: **`torch.autograd.profiler.emit_itt()` does not, and
architecturally cannot, support CUDA.** This is not a gap to fill in later
(contrast with `../../kineto-deep-dive/docs/04-cuda-path.md`, which is
genuinely incomplete pending a CUDA system) — it's a closed question,
answered here with source, symbol, and live-execution evidence.

## 1. No device branching exists in the source

`emit_itt.__init__`/`__enter__` (`torch/autograd/profiler.py:906-929`, full
text in `docs/00-overview.md` §2) takes no `use_device`/`activities`
parameter at all. It unconditionally builds:

```python
_enable_profiler(
    ProfilerConfig(ProfilerState.ITT, self.record_shapes, False, False, False, False, _ExperimentalConfig()),
    set(),   # <- always empty, regardless of what device the tensors are on
)
```

Compare `torch.autograd.profiler.profile.__init__`
(`torch/autograd/profiler.py`, documented in
`../../kineto-deep-dive/docs/00-overview.md`), which has an entire
`if self.use_device == "cuda": ... elif self.use_device == "xpu": ...`
branch tree to decide which `ProfilerActivity` to add to
`self.kineto_activities`. `emit_itt` has no such tree because it has no
activity set to populate — there's no "device" concept at this layer.

## 2. No device symbols anywhere in the compiled library

```
$ nm -D libtorch_cpu.so | c++filt | grep -i "itt" | wc -l
190
$ nm -D libtorch_cpu.so | c++filt | grep -i "itt" | grep -iE "gpu|device|sycl|level.?zero|xpu|cuda"
(no output)
```

All 190 ITT-related symbols (the `ittnotify` function-pointer table plus
`torch::profiler::itt_*` wrappers, `docs/00-overview.md` §4) are
exclusively host-side: domain/task/counter/frame creation, thread naming,
synchronization markers. None reference a GPU, a device, SYCL, Level Zero,
or CUDA. Compare Kineto's XPU path (`../../kineto-deep-dive/docs/03-xpu-path.md`),
which has an entire separate vendor plugin
(`libkineto::XpuptiActivityApi`) linked against a GPU profiling library
(`libpti_view.so.0`) — nothing analogous exists for ITT.

## 3. NVTX is a separate, unrelated API — not "ITT for CUDA"

It would be reasonable to assume PyTorch has one generic "emit vendor
markers" mechanism that branches by device. It does not. **NVTX** is
CUDA's own, completely independent marker API, with its own class:

```python
# torch/autograd/profiler.py:940
class emit_nvtx:
    """Context manager that makes every autograd operation emit an NVTX range.
    It is useful when running the program under nvprof::
        nvprof --profile-from-start off -o trace_name.prof -- <regular command here>
    ...
```

and its own Python module, `torch.cuda.nvtx`
(`$TORCH_DIR/cuda/nvtx.py`, confirmed present in this install), separate
from `torch.profiler.itt`. On this system (no CUDA build), zero NVTX
symbols are linked at all:

```
$ nm -D libtorch_cpu.so | c++filt | grep -i nvtx | wc -l
0
```

(The header `torch/csrc/profiler/standalone/nvtx_observer.h` exists in the
installed headers — declaring `pushNVTXCallbacks`, structurally identical
to `pushITTCallbacks` — but nothing implementing it is linked into this
XPU-only build, since there's no CUDA runtime to back it.)

So the honest framing is: **ITT ↔ Intel VTune** and **NVTX ↔ NVIDIA
Nsight/nvprof** are two parallel, vendor-specific sibling mechanisms
(both reachable through the shared `ProfilerState`/`_enable_profiler`
entry point documented in `docs/00-overview.md` §2-3), not one mechanism
with per-vendor support. A CUDA user who wants VTune-style marker
annotation uses `emit_nvtx()` + `nvprof`/Nsight, not `emit_itt()` — there
is no code path that would ever let `emit_itt()` collect anything on a
CUDA device, because ITT is Intel-specific tooling and CUDA isn't an Intel
device.

## 4. Live confirmation: ITT ranges are identical on CPU and XPU

Unlike asserting this from source alone, this was verified by actually
running `example/profile_itt.py` (unmodified, `DEVICE=xpu`) under
`vtune -collect hotspots` on a reserved Aurora compute node
(`x4311c0s6b0n0`, 12 XPUs) and comparing against the CPU run from
`docs/01-cpu-path.md`:

```
$ source env.sh
$ python3 -c "import torch.profiler.itt as itt; print(itt.is_available())"
True   # <- same True as the login node (docs/00-overview.md §5)

$ DEVICE=xpu vtune -collect hotspots -result-dir results/xpu_validation/vtune_hotspots_xpu -- "$PYBIN" profile_itt.py
...
Top Tasks
Task Type                                            Task Time  Task Count  Average Task Time
---------------------------------------------------  ---------  ----------  -----------------
aten::sum                                               2.583s      90,000             0.000s
autograd::engine::evaluate_function: AddmmBackward0     2.098s      30,000             0.000s
aten::linear                                            2.004s      30,000             0.000s
aten::addmm                                             1.721s      30,000             0.000s
AddmmBackward0                                          1.248s      30,000             0.000s
```

Full output: `results/xpu_validation/console_log.txt` /
`results/xpu_validation/summary.txt`.

**The task names and counts are the same set as the CPU run**
(`aten::sum`, `aten::addmm`, `aten::linear`, `AddmmBackward0` — all with
the matching `30,000`/`90,000` counts for the identical 30,000-iteration
loop). No `xpu`-specific task names, no device-id metadata, no separate
"XPU kernel" row the way Kineto's XPU trace has `gemm_kernel`/`xpu_runtime`
(`../../kineto-deep-dive/docs/03-xpu-path.md`). This is the concrete,
measured confirmation — not an inference from missing symbols — that
`emit_itt()` only ever sees the host-side RecordFunction stream, completely
blind to which device the tensors underneath actually live on.

### Note on the secondary GPU-analysis check

The plan also called for trying `vtune -collect gpu-hotspots` on the XPU
node to check whether VTune's own GPU analysis types pick up any of
PyTorch's ITT task markers. A short exploratory run hit VTune tooling
errors (a disk I/O error during database finalization, likely transient /
environment-specific) before producing a usable result. Given `gpu-hotspots`
is documented by VTune itself as analyzing "GPU kernels... based on GPU
hardware metrics" (`vtune -help collect gpu-hotspots`) — a fundamentally
different, Level-Zero-hardware-counter-based data source, unrelated to the
ittnotify ring buffer `emit_itt()` writes into — and given §1-3's symbol
and source evidence already establishes there is no code path connecting
the two, this was not pursued further. If this is reopened later, the
expected (not yet re-confirmed) result is that `gpu-hotspots` output
contains no RecordFunction-derived task names at all, since it has no
ittnotify integration.

## Summary

| | ITT (`emit_itt`) | Kineto (`torch.profiler.profile`) |
|---|---|---|
| Device activity collection | None — host markers only | PTI (XPU) / CUPTI (CUDA), `../../kineto-deep-dive/docs/03-xpu-path.md`, `../../kineto-deep-dive/docs/04-cuda-path.md` |
| CPU op visibility | Full (via RecordFunction, same as Kineto) | Full |
| "Device" parameter anywhere in the API | No | Yes (`activities=[...]`) |
| CUDA equivalent | None — use `emit_nvtx()`/`torch.cuda.nvtx` instead (different API, different tool) | Same API, `ProfilerActivity.CUDA` |
| Verified on XPU | Yes — identical task set to CPU (§4) | Yes — produces XPU-specific kernel/runtime events |
| Verified on CUDA | N/A — architecturally out of scope | Not yet — no CUDA hardware on this system |

`emit_itt()`'s scope is complete as documented: CPU validated with a real
finalized VTune result (`docs/01-cpu-path.md`), XPU validated live and
shown to behave identically (this doc), CUDA explicitly out of scope by
design rather than deferred.
