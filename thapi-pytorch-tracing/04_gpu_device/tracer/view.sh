#!/usr/bin/env bash
# View a recorded CTF trace.
#
#   source scripts/env.sh lttng
#   tracer/view.sh [trace-dir]      # defaults to newest thapi-pytorch-session-*
set -euo pipefail

TRACE_DIR="${1:-$(ls -dt "$HOME"/lttng-traces/thapi-pytorch-session-* 2>/dev/null | head -1)}"
[ -d "$TRACE_DIR" ] || { echo "ERROR: no trace dir (got '$TRACE_DIR'). Pass it explicitly." >&2; exit 1; }

babeltrace2 "$TRACE_DIR"
