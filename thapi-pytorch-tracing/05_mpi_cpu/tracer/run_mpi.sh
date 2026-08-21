#!/usr/bin/env bash
# Record ONE shared LTTng trace of an MPI (mpiexec -n N) workload under the
# RecordFunction tracer. All N ranks register with the SAME userspace session,
# so their events land in one trace, separated by LTTng's vpid context.
#
#   source <env-skill>/scripts/env.sh lttng
#   NRANKS=4 TRACER_SCOPES=function+backend BACKEND=gloo \
#     ./run_mpi.sh <label> <workload.py>
#
# Same build-once/configure-at-runtime model as the single-process skill run.sh:
# the TRACER_* toggles and workload knobs (DEVICE/BACKEND/DIM/STEPS) are read from
# the environment and PROPAGATED to every rank (Cray PALS mpiexec forwards the
# launching shell's environment). LD_PRELOAD is exported so each rank's python is
# injected.
#
# Tracer toggles: TRACER_SCOPES TRACER_THREAD TRACER_TOPLEVEL TRACER_INPUTS TRACER_SAMPLING
# Workload knobs: DEVICE BACKEND DIM STEPS N_THREADS
# Launch:         NRANKS (default 4)   TRACER_SO (default: this dir's .so)
set -o pipefail
cd "$(dirname "$0")"

LABEL="${1:?usage: run_mpi.sh <label> <workload.py>}"
WORKLOAD="${2:?usage: run_mpi.sh <label> <workload.py>}"
[ -f "$WORKLOAD" ] || { echo "ERROR: workload not found: $WORKLOAD" >&2; exit 1; }

NRANKS="${NRANKS:-4}"
PRELOAD="${TRACER_SO:-$PWD/libtorch_tracer.so}"
[ -f "$PRELOAD" ] || { echo "ERROR: tracer .so not found: $PRELOAD" >&2; exit 1; }
echo "[run] preload: $PRELOAD"
echo "[run] ranks:   $NRANKS"

# Node-local LTTNG_HOME (shared-home sessiond lock; harmless on head node).
if [ -z "${LTTNG_HOME:-}" ]; then
  export LTTNG_HOME="/tmp/lttng_${USER}_$$"
  mkdir -p "$LTTNG_HOME"
  echo "[run] LTTNG_HOME=$LTTNG_HOME"
fi

SESSION="rec-$LABEL"
lttng-sessiond --daemonize --quiet || true
lttng destroy "$SESSION" >/dev/null 2>&1 || true
lttng create "$SESSION"
lttng enable-channel --userspace --blocking-timeout=inf blocking-channel
lttng add-context --userspace --channel=blocking-channel -t vpid -t vtid
lttng enable-event --channel=blocking-channel --userspace 'lttng_ust_pytorch:*'
lttng start

# Export everything the ranks need; PALS mpiexec forwards the environment.
export LTTNG_UST_ALLOW_BLOCKING=1
export LD_PRELOAD="$PRELOAD"
export TRACER_SCOPES="${TRACER_SCOPES:-function}"
export TRACER_THREAD="${TRACER_THREAD:-global}"
export TRACER_TOPLEVEL="${TRACER_TOPLEVEL:-0}"
export TRACER_INPUTS="${TRACER_INPUTS:-0}"
export TRACER_SAMPLING="${TRACER_SAMPLING:-1.0}"
export DEVICE="${DEVICE:-cpu}"
export BACKEND="${BACKEND:-gloo}"
export DIM="${DIM:-256}"
export STEPS="${STEPS:-3}"
# torch.distributed rendezvous (gloo/ccl over TCP). Fresh port each run avoids
# EADDRINUSE from a previous run's lingering TCPStore socket.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$(( 30000 + (RANDOM % 20000) ))}"
echo "[run] rendezvous: $MASTER_ADDR:$MASTER_PORT  backend=$BACKEND device=$DEVICE"

mpiexec -n "$NRANKS" python3 "$WORKLOAD"
RC=$?

# Drop LD_PRELOAD so the post-run lttng CLI calls aren't themselves traced/injected.
unset LD_PRELOAD
lttng stop
lttng destroy "$SESSION"
TRACE_DIR="$(ls -dt "$LTTNG_HOME"/lttng-traces/${SESSION}-* 2>/dev/null | head -1)"
echo ">>> recorded session: $SESSION (workload rc=$RC)"
echo ">>> trace dir: ${TRACE_DIR:-<none found>}"
[ -n "$TRACE_DIR" ] && echo "$TRACE_DIR" > "/tmp/last_trace_${USER}.txt"
exit "$RC"
