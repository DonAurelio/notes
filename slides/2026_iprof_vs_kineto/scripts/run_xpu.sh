#!/usr/bin/env bash
#PBS -A Performance
#PBS -q debug
#PBS -l walltime=01:00:00
#PBS -l filesystems=home
#PBS -l select=1
#PBS -o /home/avivasmeza/dev/iprof_vs_kineto/results/run_xpu.out
#PBS -e /home/avivasmeza/dev/iprof_vs_kineto/results/run_xpu.err
# Drives all XPU experiment cells. Run directly on the compute node or via qsub.
set -eo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/home/avivasmeza/dev/iprof_vs_kineto
IPROF="$REPO/THAPI/build/ici/bin/iprof"
WKLD="$REPO/workloads"
TRACES="$REPO/traces/IP-XPU"
RESULTS="$REPO/results"
RAW="$RESULTS/raw.csv"

# ── Environment ───────────────────────────────────────────────────────────────
source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
module use /home/avivasmeza/thapi_env/modules
module load thapi-env/blob
module load frameworks

mkdir -p "$RESULTS" "$TRACES"

# ── Parameters ────────────────────────────────────────────────────────────────
SIZES=(256 1024 4096)
ITERS=(10 100 1000)
REPS=15
WARMUP=5

# ── Init CSV ──────────────────────────────────────────────────────────────────
[[ -f "$RAW" ]] || printf "scenario,S,N,rep,T_hot_s,T_process_s,n_events,analysis_s\n" > "$RAW"

# ── Helpers ───────────────────────────────────────────────────────────────────
TF=$(mktemp)   # /usr/bin/time output
OF=$(mktemp)   # command stdout
trap 'rm -f "$TF" "$OF"' EXIT

run_timed() {
    /usr/bin/time -f "%e" -o "$TF" "$@" > "$OF" 2>> "$RESULTS/stderr.log"
}

get_val() { grep -oP "(?<=$1=)[0-9.]+" "$OF" || echo ""; }

append_row() { printf "%s\n" "$*" | tee -a "$RAW"; }

# Sum individual-row Calls from host+device sections only (exclude Total rows and traffic).
# iprof tally has 3 sections: host (BACKEND_PYTORCH, BACKEND_ZE), device (Device profiling),
# traffic (Explicit memory traffic). Traffic counts bytes transferred, not invocations;
# Total rows double-count each section's sum. We stop at the traffic section header.
iprof_n_events() {
    awk -F'|' '
        /Explicit memory traffic/ { skip=1 }
        !skip && NF>3 {
            name=$1; gsub(/[[:space:]]/, "", name)
            calls=$4; gsub(/[[:space:]]/, "", calls)
            if (name != "Total" && name != "Name" && calls ~ /^[0-9]+$/)
                sum += calls
        }
        END { print sum+0 }
    ' "$OF"
}

# ── Pilot (interactive only) ──────────────────────────────────────────────────
if [[ "${PBS_ENVIRONMENT:-}" != "PBS_BATCH" ]]; then
    echo "=== PILOT: S=1024 N=100 3 reps ==="
    for rep in 1 2 3; do
        run_timed python3 "$WKLD/bench.py" --size 1024 --iters 100 --warmup "$WARMUP"
        echo "  BL-XPU  rep=$rep  wall_s=$(get_val wall_s)  process_s=$(cat "$TF")"

        run_timed "$IPROF" --no-analysis --trace-output "$TRACES/pilot/$rep" \
            --backends pytorch,ze -- python3 "$WKLD/bench.py" --size 1024 --iters 100 --warmup "$WARMUP"
        echo "  IP-XPU  rep=$rep  wall_s=$(get_val wall_s)  process_s=$(cat "$TF")"

        run_timed python3 "$WKLD/bench_kineto.py" --size 1024 --iters 100 --warmup "$WARMUP"
        echo "  KN-XPU  rep=$rep  wall_s=$(get_val wall_s)  n_events=$(get_val n_events)  process_s=$(cat "$TF")"
    done
    echo ""
    echo "Review pilot times above. Press Enter to run full matrix, or Ctrl-C to abort."
    read -r
else
    echo "=== Batch mode: skipping interactive pilot (see pilot_results.md) ==="
fi

# ── Phase 1: collection ───────────────────────────────────────────────────────
echo "=== Phase 1: collection ==="
for S in "${SIZES[@]}"; do
    for N in "${ITERS[@]}"; do
        [[ $S -eq 4096 && $N -eq 1000 ]] && continue
        echo "  S=$S N=$N"
        for rep in $(seq 1 "$REPS"); do

            run_timed python3 "$WKLD/bench.py" --size "$S" --iters "$N" --warmup "$WARMUP"
            append_row "BL-XPU,$S,$N,$rep,$(get_val wall_s),$(cat "$TF"),,"

            mkdir -p "$TRACES/${S}_${N}"
            run_timed "$IPROF" --no-analysis --trace-output "$TRACES/${S}_${N}/$rep" \
                --backends pytorch,ze -- python3 "$WKLD/bench.py" --size "$S" --iters "$N" --warmup "$WARMUP"
            append_row "IP-XPU,$S,$N,$rep,$(get_val wall_s),$(cat "$TF"),,"

            run_timed python3 "$WKLD/bench_kineto.py" --size "$S" --iters "$N" --warmup "$WARMUP"
            append_row "KN-XPU,$S,$N,$rep,$(get_val wall_s),$(cat "$TF"),$(get_val n_events),$(get_val analysis_s)"

        done
    done
done

# ── Phase 2: iprof replay ─────────────────────────────────────────────────────
echo "=== Phase 2: iprof replay ==="
TMPCSV=$(mktemp)
cp "$RAW" "$TMPCSV"

for S in "${SIZES[@]}"; do
    for N in "${ITERS[@]}"; do
        [[ $S -eq 4096 && $N -eq 1000 ]] && continue
        for rep in $(seq 1 "$REPS"); do
            trace="$TRACES/${S}_${N}/$rep"

            /usr/bin/time -f "%e" -o "$TF" "$IPROF" --replay "$trace" > "$OF" 2>>"$RESULTS/stderr.log"
            analysis_s=$(cat "$TF")
            n_events=$(iprof_n_events)

            echo "  IP-XPU S=$S N=$N rep=$rep  n_events=$n_events  analysis_s=$analysis_s"

            awk -F',' -v s="$S" -v n="$N" -v r="$rep" -v ne="$n_events" -v as="$analysis_s" \
                'BEGIN{OFS=","} $1=="IP-XPU" && $2==s && $3==n && $4==r {$7=ne; $8=as} 1' \
                "$TMPCSV" > "$RAW"
            cp "$RAW" "$TMPCSV"
        done
    done
done
rm -f "$TMPCSV"

echo "=== Done. Results in $RAW ==="

python3 "$REPO/scripts/summarize.py"
python3 "$REPO/analysis/plot_overhead.py"
echo "=== summary.csv ==="
cat "$RESULTS/summary.csv"
