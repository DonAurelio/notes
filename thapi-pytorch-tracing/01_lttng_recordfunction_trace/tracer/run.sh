#!/usr/bin/env bash
# Record a CTF trace of an UNMODIFIED torch workload (tracing injected via
# LD_PRELOAD, no Python edits).
#
#   source scripts/env.sh lttng
#   tracer/run.sh [workload.py]
set -euo pipefail
cd "$(dirname "$0")"

PRELOAD="$PWD/libtorch_tracer.so"
WORKLOAD="${1:-$PWD/../examples/model.py}"
SESSION="thapi-pytorch-session"

lttng-sessiond --daemonize --quiet || true
lttng destroy "$SESSION" >/dev/null 2>&1 || true
lttng create "$SESSION"

# Blocking channel so a short run drops no events; vpid/vtid for pairing/nesting.
lttng enable-channel --userspace --blocking-timeout=inf blocking-channel
lttng add-context --userspace --channel=blocking-channel -t vpid -t vtid
lttng enable-event --channel=blocking-channel --userspace 'lttng_ust_pytorch:*'
lttng start

LTTNG_UST_ALLOW_BLOCKING=1 LD_PRELOAD="$PRELOAD" python3 "$WORKLOAD"

lttng stop
lttng destroy "$SESSION"
