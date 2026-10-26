#!/usr/bin/env bash
#PBS -A Performance
#PBS -q debug
#PBS -l walltime=01:00:00
#PBS -l filesystems=home
#PBS -l select=1
#PBS -o /home/avivasmeza/dev/iprof_vs_kineto/exp_v1.2/results/run_xpu_4096_1000.out
#PBS -e /home/avivasmeza/dev/iprof_vs_kineto/exp_v1.2/results/run_xpu_4096_1000.err

set -eo pipefail

REPO=/home/avivasmeza/dev/iprof_vs_kineto
EXP=$REPO/exp_v1.2
IPROF="$REPO/THAPI/build/ici/bin/iprof"
WKLD=$EXP/workloads
TRACES=$EXP/traces/IP-XPU
RESULTS=$EXP/results
RAW=$RESULTS/raw.csv

source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
module use /home/avivasmeza/thapi_env/modules
module load thapi-env/blob
module load frameworks

mkdir -p "$RESULTS" "$TRACES"

S=4096
N=1000
REPS=15
WARMUP=5

[[ -f "$RAW" ]] || printf "scenario,S,N,rep,T_hot_s,T_process_s,n_events,analysis_s\n" > "$RAW"

TF=$(mktemp); OF=$(mktemp)
trap 'rm -f "$TF" "$OF"' EXIT

run_timed() { /usr/bin/time -f "%e" -o "$TF" "$@" > "$OF" 2>>"$RESULTS/stderr.log"; }
get_val()   { grep -oP "(?<=$1=)[0-9.]+" "$OF" || echo ""; }
append_row(){ printf "%s\n" "$*" | tee -a "$RAW"; }

echo "=== S=$S N=$N ==="
mkdir -p "$TRACES/${S}_${N}"
for rep in $(seq 1 "$REPS"); do

    # BL
    run_timed python3 "$WKLD/bench.py" --size "$S" --iters "$N" --warmup "$WARMUP"
    append_row "BL-XPU,$S,$N,$rep,$(get_val wall_s),$(cat "$TF"),,"

    # IP: collect → count raw events → replay → complete row
    run_timed "$IPROF" --no-analysis --backends pytorch,ze \
        --trace-output "$TRACES/${S}_${N}/$rep" \
        -- python3 "$WKLD/bench.py" --size "$S" --iters "$N" --warmup "$WARMUP"
    T_hot=$(get_val wall_s)
    T_process=$(cat "$TF")
    n_events=$(babeltrace2 "$TRACES/${S}_${N}/$rep" | wc -l)
    /usr/bin/time -f "%e" -o "$TF" \
        "$IPROF" --replay "$TRACES/${S}_${N}/$rep" > /dev/null 2>>"$RESULTS/stderr.log"
    append_row "IP-XPU,$S,$N,$rep,$T_hot,$T_process,$n_events,$(cat "$TF")"

    # KN
    run_timed python3 "$WKLD/bench_kineto.py" --size "$S" --iters "$N" --warmup "$WARMUP"
    append_row "KN-XPU,$S,$N,$rep,$(get_val wall_s),$(cat "$TF"),$(get_val n_events),$(get_val analysis_s)"

done

echo "=== Summarizing ==="
python3 "$EXP/scripts/summarize.py"
echo "=== Done. Results in $RESULTS ==="
