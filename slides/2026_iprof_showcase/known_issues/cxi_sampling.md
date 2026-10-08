# Resolved: CXI sampling (`iprof -s`) appeared to produce no events

**Status:** resolved, not a THAPI bug. Moved back into the main showcase
(`../README.md`, section 2.3 "Performance Counter Sampling (CXI)").

## What looked broken

`iprof -s --analysis-output iprof_summary.txt -- python model.py` showed
no CXI counter data anywhere: not in the tally, not in the raw trace.
`THAPI_SAMPLING_LIBRARIES` and `LTTNG_UST_CXI_SAMPLING_CXI` were both
correctly set ([`debug_env_dump.txt`](debug_env_dump.txt)), and the
`lttng_ust_cxi_sampling` tracepoint was enabled for the session, on a
node with real CXI hardware and real telemetry counters confirmed
readable directly. [`iprof_summary.txt`](iprof_summary.txt) and
[`model.py`](model.py) are the run that first surfaced this.

## What was actually happening

Two separate mistakes stacked up, not a bug in the CXI plugin:

1. **Looking in the wrong view.** `--analysis-output` (the tally) only
   aggregates named, duration-based calls. CXI counter samples are
   periodic, not call-scoped, so THAPI's `to_aggreg` step doesn't carry
   them into the tally at all, by design. They only ever show up in the
   raw trace, the interval view, or the Perfetto timeline.
2. **Requesting the default backend set with `-s`.** The default
   `--backends` list includes `ze`, so any `-s` run also tries to load
   `libZESampling.so`. On this system that crashes
   `thapi_sampling_daemon` outright
   (`libffi.so.8: cannot open shared object file`), which silently drops
   *every* sampling plugin for that run, CXI included, with no error
   surfaced back through `iprof`. Restricting to
   `--backend cxi,pytorch` avoids loading the ZE plugin and the daemon
   starts cleanly.

Once both were corrected (check the raw/interval/timeline views, and
scope `--backend` to just what's needed), CXI sampling works as
documented: `lttng_ust_cxi_sampling:cxi` events appear in the raw trace,
and `sampling:nic` rows appear in the interval view, ticking at the
expected ~100ms period.

## How this was confirmed

Reproduced the exact pattern from THAPI's own integration test,
[`integration_tests/sampling.bats`](https://github.com/argonne-lcf/THAPI/blob/devel/integration_tests/sampling.bats)
(`sampling_cxi`), which always runs with `--backend cxi` (never the
default set) and `--no-analysis` (it reads the raw trace directly with
`babeltrace_thapi`, never the tally). Running the project's own
`model.py` the same way, restricted to `--backend cxi,pytorch`, produced
CXI samples immediately, on both a short run and a longer `sleep 2`
control matching the bats test's own timing. The earlier failed attempts
all used the default backend set (which includes `ze`) and read from the
tally; neither reproduces with those two mistakes removed.

## Where it was tested

This project's own THAPI build: `/home/avivasmeza/dev/shared`, same
commit as the `pytorch-pretty-print` branch. Tested on Aurora compute
nodes with 8 CXI NICs present (`/sys/class/cxi/cxi0` through `cxi7`).
