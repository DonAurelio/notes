#!/usr/bin/env bash
# Convert a recorded CTF trace to readable text with babeltrace2.
#
#   source <env-skill>/scripts/env.sh lttng
#   ./view.sh [trace-dir]      # defaults to the last session run.sh recorded
#
# With no argument, uses the trace dir run.sh stashed in /tmp/last_trace_$USER.txt,
# falling back to the newest rec-* session under $LTTNG_HOME (or $HOME).
set -euo pipefail

TRACE_DIR="${1:-}"
if [ -z "$TRACE_DIR" ]; then
  if [ -f "/tmp/last_trace_${USER}.txt" ]; then
    TRACE_DIR="$(cat "/tmp/last_trace_${USER}.txt")"
  else
    ROOT="${LTTNG_HOME:-$HOME}"
    TRACE_DIR="$(ls -dt "$ROOT"/lttng-traces/rec-* 2>/dev/null | head -1)"
  fi
fi
[ -d "$TRACE_DIR" ] || { echo "ERROR: no trace dir (got '$TRACE_DIR'). Pass it explicitly." >&2; exit 1; }

babeltrace2 "$TRACE_DIR"
