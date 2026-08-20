#!/usr/bin/env bash
# Record ONE LTTng trace of a workload under the RecordFunction tracer.
#
#   source <env-skill>/scripts/env.sh lttng     # torch + LTTng tooling
#   ./run.sh <label> <workload.py>              # -> a named LTTng session
#
# The tracer .so is build-once/configure-at-runtime: set the TRACER_* toggles in
# the environment before calling to change WHAT is captured (no rebuild). Workload
# knobs (DEVICE/SYNC/N_THREADS/DIM/STEPS) are also passed through from the env.
#
# Tracer toggles (see the tracer-build skill):
#   TRACER_SCOPES   function | function+backward   (aspect 2)
#   TRACER_THREAD   global | local                 (aspect 3)
#   TRACER_TOPLEVEL 0 | 1                           (aspect 4)
#   TRACER_INPUTS   0 | 1                           (aspect 5)
#   TRACER_SAMPLING 0..1                            (aspect 6)
# Workload knobs (passed straight through to python):
#   DEVICE cpu|xpu   SYNC 0|1   N_THREADS   DIM   STEPS
#
# TRACER_SO   path to libtorch_tracer.so (default: the tracer-build skill's copy).
set -o pipefail   # NOT -u: env.sh references $1 when sourced
cd "$(dirname "$0")"

LABEL="${1:?usage: run.sh <label> <workload.py>}"
WORKLOAD="${2:?usage: run.sh <label> <workload.py>}"
[ -f "$WORKLOAD" ] || { echo "ERROR: workload not found: $WORKLOAD" >&2; exit 1; }

# Locate the tracer .so. Default to the sibling tracer-build skill's build output.
DEFAULT_SO="$HOME/.claude/skills/pytorch-basic-tracing-tracer-build/scripts/libtorch_tracer.so"
PRELOAD="${TRACER_SO:-$DEFAULT_SO}"
[ -f "$PRELOAD" ] || { echo "ERROR: tracer .so not found: $PRELOAD" >&2
  echo "       build it first (pytorch-basic-tracing-tracer-build) or set TRACER_SO." >&2; exit 1; }
echo "[run] preload: $PRELOAD"

# Node-local LTTNG_HOME: on a compute node the shared \$HOME/.lttng holds a
# sessiond lock from the login node, so a compute-node sessiond can't start
# ("A session daemon is already running"). Isolate the daemon + raw CTF to
# node-local /tmp. Harmless on the head node too. Only override if unset.
if [ -z "${LTTNG_HOME:-}" ]; then
  export LTTNG_HOME="/tmp/lttng_${USER}_$$"
  mkdir -p "$LTTNG_HOME"
  echo "[run] LTTNG_HOME=$LTTNG_HOME (node-local; avoids shared-home sessiond lock)"
fi

SESSION="rec-$LABEL"
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
RC=$?

lttng stop
lttng destroy "$SESSION"
TRACE_DIR="$(ls -dt "$LTTNG_HOME"/lttng-traces/${SESSION}-* 2>/dev/null | head -1)"
echo ">>> recorded session: $SESSION (workload rc=$RC)"
echo ">>> trace dir: ${TRACE_DIR:-<none found>}"
[ -n "$TRACE_DIR" ] && echo "$TRACE_DIR" > "/tmp/last_trace_${USER}.txt"
exit "$RC"
