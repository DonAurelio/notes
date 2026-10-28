#!/usr/bin/env bash
#PBS -A Performance
#PBS -q debug
#PBS -l walltime=01:00:00
#PBS -l filesystems=home
#PBS -l select=2
#PBS -o /home/avivasmeza/dev/iprof_vs_kineto/notes/slides/2026_iprof_vs_kineto_ddp/results/run_ddp.out
#PBS -e /home/avivasmeza/dev/iprof_vs_kineto/notes/slides/2026_iprof_vs_kineto_ddp/results/run_ddp.err

set -eo pipefail

REPO=/home/avivasmeza/dev/iprof_vs_kineto
EXP=$REPO/notes/slides/2026_iprof_vs_kineto_ddp
IPROF="$REPO/THAPI/build/ici/bin/iprof"
WKLD=$EXP/workloads
SCRATCH=$REPO/scratch_dir/full
RESULTS=$EXP/results
RAW=$RESULTS/raw.csv

source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
module use /home/avivasmeza/thapi_env/modules
module load thapi-env/blob
module load frameworks

mkdir -p "$RESULTS" "$SCRATCH"

# Read the two allocated hostnames from PBS_NODEFILE
mapfile -t HOSTS < <(sort -u "$PBS_NODEFILE")
HOST0=${HOSTS[0]}
HOST1=${HOSTS[1]}
export MASTER_ADDR=$HOST0
export PALS_WORLD_SIZE=2

STEPS=10        # hot-loop steps; adjust after smoke test
REPS=10         # repetitions per scenario
MODEL_ARGS="--device xpu --steps $STEPS --batch-size 1 --seq-len 16 --dim 64 --layers 1 --heads 4 --vocab-size 128 --bf16"
MPIEXEC="mpiexec --no-transfer -n 2 -ppn 1 -hosts $HOST0,$HOST1"

TF=$(mktemp); OF=$(mktemp)
trap 'rm -f "$TF" "$OF"' EXIT

run_timed()       { /usr/bin/time -f "%e" -o "$TF" "$@" > "$OF" 2>>"$RESULTS/stderr.log"; }
run_timed_iprof() { /usr/bin/time -f "%e" -o "$TF" "$@" > "$OF" 2>>"$RESULTS/stderr.log" || true; }
get_val()         { grep -oP "(?<=$1=)[0-9.]+" "$OF" || echo ""; }
append_row(){ printf "%s\n" "$*" | tee -a "$RAW"; }

[[ -f "$RAW" ]] || printf "scenario,rep,T_hot_s,T_process_s,n_events,analysis_s\n" > "$RAW"

echo "Hosts: $HOST0 $HOST1  STEPS=$STEPS  REPS=$REPS"

for rep in $(seq 1 "$REPS"); do
    echo "=== rep $rep / $REPS ==="

    # BL
    run_timed $MPIEXEC "$WKLD/ccl_local_wrap.sh" \
        python3 "$WKLD/train_llama3_demo.py" $MODEL_ARGS
    append_row "BL,$rep,$(get_val wall_s),$(cat "$TF"),,"

    # IP
    mkdir -p "$SCRATCH/IP"
    run_timed_iprof $MPIEXEC "$WKLD/ccl_local_wrap.sh" \
        "$IPROF" --no-analysis --backends pytorch,ze \
        --trace-output "$SCRATCH/IP/rep${rep}" \
        -- python3 "$WKLD/train_llama3_demo.py" $MODEL_ARGS
    T_hot_ip=$(get_val wall_s)
    T_proc_ip=$(cat "$TF")
    n_events_ip=$(babeltrace2 "$SCRATCH/IP/rep${rep}" | wc -l)
    /usr/bin/time -f "%e" -o "$TF" \
        "$IPROF" --replay "$SCRATCH/IP/rep${rep}" > /dev/null 2>>"$RESULTS/stderr.log"
    append_row "IP,$rep,$T_hot_ip,$T_proc_ip,$n_events_ip,$(cat "$TF")"

    # KN
    run_timed $MPIEXEC "$WKLD/ccl_local_wrap.sh" \
        python3 "$WKLD/train_llama3_kineto.py" $MODEL_ARGS
    append_row "KN,$rep,$(get_val wall_s),$(cat "$TF"),$(get_val n_events),$(get_val analysis_s)"

done

echo "=== Summarizing ==="
python3 "$EXP/scripts/summarize_ddp.py"
echo "=== Done. Results in $RESULTS ==="
