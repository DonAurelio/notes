#!/usr/bin/env bash
#PBS -A Performance
#PBS -q debug
#PBS -l walltime=00:20:00
#PBS -l filesystems=home
#PBS -l select=1
#PBS -o /home/avivasmeza/dev/iprof_vs_kineto/results/replay_only.out
#PBS -e /home/avivasmeza/dev/iprof_vs_kineto/results/replay_only.err
# Re-runs iprof --replay for all 8 cells x 15 reps on a compute node.
# Updates n_events and analysis_s in raw.csv without touching BL/KN rows
# or Phase 1 timings. Submit with: qsub scripts/replay_only.sh
set -eo pipefail

REPO=/home/avivasmeza/dev/iprof_vs_kineto
IPROF="$REPO/THAPI/build/ici/bin/iprof"
TRACES="$REPO/traces/IP-XPU"
RESULTS="$REPO/results"
RAW="$RESULTS/raw_v2.csv"        # never touches raw.csv

source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
module use /home/avivasmeza/thapi_env/modules
module load thapi-env/blob
module load frameworks

SIZES=(256 1024 4096)
ITERS=(10 100 1000)
REPS=15

TF=$(mktemp)
OF=$(mktemp)
trap 'rm -f "$TF" "$OF"' EXIT

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

echo "=== replay_only.sh: Phase 2 replay on compute node ==="
# Seed raw_v2.csv from the original (preserves BL/KN rows and Phase 1 timings)
cp "$RESULTS/raw.csv" "$RAW"

TMPCSV=$(mktemp)
cp "$RAW" "$TMPCSV"

for S in "${SIZES[@]}"; do
    for N in "${ITERS[@]}"; do
        [[ $S -eq 4096 && $N -eq 1000 ]] && continue
        for rep in $(seq 1 "$REPS"); do
            trace="$TRACES/${S}_${N}/$rep"
            if [[ ! -d "$trace" ]]; then
                echo "  MISSING trace S=$S N=$N rep=$rep — skipping"
                continue
            fi

            /usr/bin/time -f "%e" -o "$TF" "$IPROF" --replay "$trace" \
                > "$OF" 2>>"$RESULTS/replay_only_stderr.log"
            analysis_s=$(cat "$TF")
            n_events=$(iprof_n_events)

            echo "  S=$S N=$N rep=$rep  n_events=$n_events  analysis_s=$analysis_s"

            awk -F',' -v s="$S" -v n="$N" -v r="$rep" -v ne="$n_events" -v as="$analysis_s" \
                'BEGIN{OFS=","} $1=="IP-XPU" && $2==s && $3==n && $4==r {$7=ne; $8=as} 1' \
                "$TMPCSV" > "$RAW"
            cp "$RAW" "$TMPCSV"
        done
    done
done
rm -f "$TMPCSV"

echo "=== Rerunning summarize + plot (v2) ==="
python3 "$REPO/scripts/summarize.py" \
    --raw "$RESULTS/raw_v2.csv" \
    --out "$RESULTS/summary_v2.csv"
python3 "$REPO/analysis/plot_overhead.py" \
    --summary "$RESULTS/summary_v2.csv" \
    --out     "$REPO/analysis/overhead_v2.png"

echo "=== summary_v2.csv ==="
cat "$RESULTS/summary_v2.csv"
echo "=== replay_only DONE ==="
