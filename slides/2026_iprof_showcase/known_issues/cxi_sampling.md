# Known issue: CXI sampling (`iprof -s`) produces no events

**Status:** unresolved. Excluded from the main showcase
(`../README.md`) until fixed. When it's working, this content should move
back into "Features Showcase" as its own subsection, per the original
outline (between "Programming-Model-Based Tracing" and "Intel ITT
Backend").

## What should work

`-s`/`--sample` turns on THAPI's sampling daemon, which reads hardware
counters on a fixed interval, independent of the workload's own
instrumented calls. One of its plugins samples real CXI (Slingshot NIC)
telemetry counters under `/sys/class/cxi/*/device/telemetry`.

```bash
iprof -s --analysis-output iprof_summary.txt -- python model.py
```

```python
# model.py
import torch
x = torch.randn(2048, 2048, device="xpu")
y = torch.matmul(x, x)
torch.xpu.synchronize()
```

## What actually happens

No CXI counter data appears anywhere in the output (not in the tally, not
in the raw trace). Confirmed with `--debug 0` that `iprof` correctly
assembles everything the CXI sampling plugin needs:

```text
THAPI_SAMPLING_LIBRARIES => [".../ze/libZESampling.so", ".../cxi/libCXISampling.so", ".../sampling/libHeartbeatSampling.so"]
LTTNG_UST_CXI_SAMPLING_CXI => 1
```

The plugin library is loaded, its activation switch is set, and the
`lttng_ust_cxi_sampling` tracepoint is enabled for the session, but no
`lttng_ust_cxi_sampling` events are ever emitted in the resulting trace,
even on a node with real CXI hardware (`/sys/class/cxi/cxi0..7`, real
non-zero telemetry counters confirmed readable directly).

**The sampling daemon mechanism itself is confirmed functional**: the ZE
sampling plugin, activated the same way by the same `-s` flag, does work
(its `lttng_ust_ze_sampling:deviceProperties` events appear correctly in
the trace). The gap is specific to the CXI plugin.

## Where it was tested

This project's own THAPI build: `~/dev/pr0/THAPI`, branch
`pytorch-pretty-print`, binary at `~/dev/pr0/THAPI/build/ici/bin/iprof`.
Tested on an Aurora compute node with 8 CXI NICs present
(`/sys/class/cxi/cxi0` through `cxi7`).

## Saved evidence

- [`debug_env_dump.txt`](debug_env_dump.txt) — `iprof --debug 0 -s -- true`
  output showing the correctly-assembled `THAPI_SAMPLING_LIBRARIES` and
  `LTTNG_UST_CXI_SAMPLING_CXI` env vars.
- [`iprof_summary.txt`](iprof_summary.txt) — tally from a real run with
  `-s` enabled; no CXI section appears.
- [`model.py`](model.py) — the base code used.

## Next steps

This looks like a bug in THAPI itself (`backends/cxi/cxi_sampling_plugin.c`
and/or `xprof/xprof.rb.in`'s sampling-daemon wiring), not a usage error on
our end. To debug further: check whether the CXI plugin's
`thapi_initialize_sampling_plugin()` actually runs inside the daemon
process (e.g. by having it log to stderr or a file on entry), since the
daemon communicates with its parent over a pipe handshake
(`thapi_sampling_daemon.cpp`) that makes it awkward to run standalone for
isolated testing.
