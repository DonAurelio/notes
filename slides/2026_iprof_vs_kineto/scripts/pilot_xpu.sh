#!/usr/bin/env bash
#PBS -A Performance
#PBS -q debug
#PBS -l walltime=00:20:00
#PBS -l filesystems=home
#PBS -l select=1
#PBS -o /home/avivasmeza/dev/iprof_vs_kineto/results/pilot.out
#PBS -e /home/avivasmeza/dev/iprof_vs_kineto/results/pilot.err
set -eo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/home/avivasmeza/dev/iprof_vs_kineto
IPROF="$REPO/THAPI/build/ici/bin/iprof"
WKLD="$REPO/workloads"
TRACES="$REPO/traces/IP-XPU/pilot"
RESULTS="$REPO/results"
RAW="$RESULTS/raw.csv"

# ── Environment ───────────────────────────────────────────────────────────────
source /usr/share/lmod/lmod/init/bash
module use /home/avivasmeza/thapi_env/modules
module load thapi-env/blob
module load frameworks

# Clean pilot traces from prior runs so iprof can create fresh dirs
rm -rf "$TRACES"
mkdir -p "$RESULTS" "$(dirname "$TRACES")"

# ── Always start fresh for the pilot ─────────────────────────────────────────
printf "scenario,S,N,rep,T_hot_s,T_process_s,n_events,analysis_s\n" > "$RAW"

# ── Config ────────────────────────────────────────────────────────────────────
S=1024; N=100; WARMUP=5; REPS=3

# ── Helpers ───────────────────────────────────────────────────────────────────
TF=$(mktemp); OF=$(mktemp)
trap 'rm -f "$TF" "$OF"' EXIT

run_timed() {
    /usr/bin/time -f "%e" -o "$TF" "$@" > "$OF" 2>> "$RESULTS/pilot_stderr.log"
}

get_val() { grep -oP "(?<=$1=)[0-9.]+" "$OF" || echo ""; }

append_row() { printf "%s\n" "$*" | tee -a "$RAW"; }

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

# ── Collection ────────────────────────────────────────────────────────────────
echo "=== PILOT collection: S=$S N=$N REPS=$REPS ==="
for rep in $(seq 1 $REPS); do
    echo "  rep=$rep"

    run_timed python3 "$WKLD/bench.py" --size "$S" --iters "$N" --warmup "$WARMUP"
    echo "    BL-XPU  wall_s=$(get_val wall_s)  process_s=$(cat "$TF")"
    append_row "BL-XPU,$S,$N,$rep,$(get_val wall_s),$(cat "$TF"),,"

    run_timed "$IPROF" --no-analysis --trace-output "$TRACES/$rep" \
        --backends pytorch,ze -- python3 "$WKLD/bench.py" --size "$S" --iters "$N" --warmup "$WARMUP"
    echo "    IP-XPU  wall_s=$(get_val wall_s)  process_s=$(cat "$TF")"
    append_row "IP-XPU,$S,$N,$rep,$(get_val wall_s),$(cat "$TF"),,"

    run_timed python3 "$WKLD/bench_kineto.py" --size "$S" --iters "$N" --warmup "$WARMUP"
    echo "    KN-XPU  wall_s=$(get_val wall_s)  n_events=$(get_val n_events)  process_s=$(cat "$TF")"
    append_row "KN-XPU,$S,$N,$rep,$(get_val wall_s),$(cat "$TF"),$(get_val n_events),$(get_val analysis_s)"
done

# ── Replay ────────────────────────────────────────────────────────────────────
echo "=== PILOT replay ==="
TMPCSV=$(mktemp)
cp "$RAW" "$TMPCSV"

for rep in $(seq 1 $REPS); do
    /usr/bin/time -f "%e" -o "$TF" "$IPROF" --replay "$TRACES/$rep" > "$OF" 2>>"$RESULTS/pilot_stderr.log"
    analysis_s=$(cat "$TF")
    n_events=$(iprof_n_events)
    echo "  rep=$rep  n_events=$n_events  analysis_s=$analysis_s"

    awk -F',' -v s="$S" -v n="$N" -v r="$rep" -v ne="$n_events" -v as="$analysis_s" \
        'BEGIN{OFS=","} $1=="IP-XPU" && $2==s && $3==n && $4==r {$7=ne; $8=as} 1' \
        "$TMPCSV" > "$RAW"
    cp "$RAW" "$TMPCSV"
done
rm -f "$TMPCSV"

# ── Pipeline ──────────────────────────────────────────────────────────────────
echo "=== Pipeline ==="
python3 "$REPO/scripts/summarize.py"
python3 "$REPO/analysis/plot_overhead.py"

echo "=== raw.csv ==="
cat "$RAW"
echo "=== summary.csv ==="
cat "$RESULTS/summary.csv"
echo "=== PILOT DONE ==="
