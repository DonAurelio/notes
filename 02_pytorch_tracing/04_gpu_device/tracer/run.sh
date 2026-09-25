#!/usr/bin/env bash
# Step 3 record recipe. Passes the aspect toggles through to the tracer and
# writes each experiment to its own named LTTng session so traces don't mix.
#
#   source scripts/env.sh lttng
#   tracer_step3/run.sh <label> <workload.py>
#
# Toggles (set in the environment before calling, e.g. TRACER_SCOPES=...):
#   TRACER_SCOPES   function | function+backward   (aspect 2)
#   TRACER_THREAD   global | local                 (aspect 3)
#   TRACER_TOPLEVEL 0 | 1                           (aspect 4)
#   TRACER_INPUTS   0 | 1                           (aspect 5)
#   TRACER_SAMPLING 0..1                            (aspect 6)
set -euo pipefail
cd "$(dirname "$0")"

LABEL="${1:?usage: run.sh <label> <workload.py>}"
WORKLOAD="${2:?usage: run.sh <label> <workload.py>}"
PRELOAD="$PWD/libtorch_tracer.so"
SESSION="step3-$LABEL"

lttng-sessiond --daemonize --quiet || true
lttng destroy "$SESSION" >/dev/null 2>&1 || true
lttng create "$SESSION"
lttng enable-channel --userspace --blocking-timeout=inf blocking-channel
lttng add-context --userspace --channel=blocking-channel -t vpid -t vtid
lttng enable-event --channel=blocking-channel --userspace 'lttng_ust_pytorch:*'
lttng start

LTTNG_UST_ALLOW_BLOCKING=1 \
  TRACER_SCOPES="${TRACER_SCOPES:-function}" \
  TRACER_THREAD="${TRACER_THREAD:-global}" \
  TRACER_TOPLEVEL="${TRACER_TOPLEVEL:-0}" \
  TRACER_INPUTS="${TRACER_INPUTS:-0}" \
  TRACER_SAMPLING="${TRACER_SAMPLING:-1.0}" \
  LD_PRELOAD="$PRELOAD" python3 "$WORKLOAD"

lttng stop
lttng destroy "$SESSION"
echo ">>> recorded session: $SESSION"
