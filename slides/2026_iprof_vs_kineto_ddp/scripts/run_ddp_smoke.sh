#!/usr/bin/env bash
#PBS -A Performance
#PBS -q debug
#PBS -l walltime=00:30:00
#PBS -l filesystems=home
#PBS -l select=2
#PBS -o /home/avivasmeza/dev/iprof_vs_kineto/notes/slides/2026_iprof_vs_kineto_ddp/results/smoke.out
#PBS -e /home/avivasmeza/dev/iprof_vs_kineto/notes/slides/2026_iprof_vs_kineto_ddp/results/smoke.err

set -eo pipefail

mapfile -t HOSTS < <(sort -u "$PBS_NODEFILE")
HOST0=${HOSTS[0]}
HOST1=${HOSTS[1]}

REPO=/home/avivasmeza/dev/iprof_vs_kineto
EXP=$REPO/notes/slides/2026_iprof_vs_kineto_ddp
IPROF="$REPO/THAPI/build/ici/bin/iprof"
WKLD=$EXP/workloads
SCRATCH=$REPO/scratch_dir/smoke
RESULTS=$EXP/results
RAW=$RESULTS/raw.csv

source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
module use /home/avivasmeza/thapi_env/modules
module load thapi-env/blob
module load frameworks

mkdir -p "$RESULTS" "$SCRATCH"

STEPS=5
MODEL_ARGS="--device xpu --steps $STEPS --batch-size 1 --seq-len 16 --dim 64 --layers 1 --heads 4 --vocab-size 128 --bf16"
MPIEXEC="mpiexec --no-transfer -n 2 -ppn 1 -hosts $HOST0,$HOST1"

export MASTER_ADDR=$HOST0
export PALS_WORLD_SIZE=2

TF=$(mktemp); OF=$(mktemp); EF=$(mktemp)
trap 'rm -f "$TF" "$OF" "$EF"' EXIT

run_timed()      { /usr/bin/time -f "%e" -o "$TF" "$@" > "$OF" 2>"$EF" || { cat "$EF" >&2; exit 1; }; }
run_timed_iprof() { /usr/bin/time -f "%e" -o "$TF" "$@" > "$OF" 2>"$EF" || true; }
get_val()        { grep -oP "(?<=$1=)[0-9.]+" "$OF" || echo ""; }

[[ -f "$RAW" ]] || printf "scenario,rep,T_hot_s,T_process_s,n_events,analysis_s\n" > "$RAW"

echo "=== [BL] baseline ==="
run_timed $MPIEXEC "$WKLD/ccl_local_wrap.sh" \
    python3 "$WKLD/train_llama3_demo.py" $MODEL_ARGS
T_hot_bl=$(get_val wall_s)
T_proc_bl=$(cat "$TF")
printf "BL,1,%s,%s,,\n" "$T_hot_bl" "$T_proc_bl" | tee -a "$RAW"
echo "  T_hot=$T_hot_bl  T_process=$T_proc_bl"

echo "=== [IP] iprof ==="
mkdir -p "$SCRATCH/IP"
run_timed_iprof $MPIEXEC "$WKLD/ccl_local_wrap.sh" \
    "$IPROF" --no-analysis --backends pytorch,ze \
    --trace-output "$SCRATCH/IP/rep1" \
    -- python3 "$WKLD/train_llama3_demo.py" $MODEL_ARGS
T_hot_ip=$(get_val wall_s)
T_proc_ip=$(cat "$TF")
n_events_ip=$(babeltrace2 "$SCRATCH/IP/rep1" | wc -l)
/usr/bin/time -f "%e" -o "$TF" \
    "$IPROF" --replay "$SCRATCH/IP/rep1" > /dev/null 2>>"$RESULTS/stderr.log"
analysis_ip=$(cat "$TF")
printf "IP,1,%s,%s,%s,%s\n" "$T_hot_ip" "$T_proc_ip" "$n_events_ip" "$analysis_ip" | tee -a "$RAW"
echo "  T_hot=$T_hot_ip  T_process=$T_proc_ip  n_events=$n_events_ip  analysis_s=$analysis_ip"

echo "=== [KN] Kineto ==="
run_timed $MPIEXEC "$WKLD/ccl_local_wrap.sh" \
    python3 "$WKLD/train_llama3_kineto.py" $MODEL_ARGS
T_hot_kn=$(get_val wall_s)
T_proc_kn=$(cat "$TF")
n_events_kn=$(get_val n_events)
analysis_kn=$(get_val analysis_s)
printf "KN,1,%s,%s,%s,%s\n" "$T_hot_kn" "$T_proc_kn" "$n_events_kn" "$analysis_kn" | tee -a "$RAW"
echo "  T_hot=$T_hot_kn  T_process=$T_proc_kn  n_events=$n_events_kn  analysis_s=$analysis_kn"
echo "--- KN stdout ---"
cat "$OF"
echo "--- end KN stdout ---"

echo ""
echo "=== Preliminary Report (1 rep, steps=$STEPS) ==="
printf "%-12s %-12s %-14s %-12s %-14s\n" "Scenario" "T_hot (s)" "T_process (s)" "n_events" "analysis (s)"
printf "%-12s %-12s %-14s %-12s %-14s\n" "--------" "---------" "-------------" "--------" "------------"
printf "%-12s %-12s %-14s %-12s %-14s\n" "BL"  "$T_hot_bl"  "$T_proc_bl"  "-"           "-"
printf "%-12s %-12s %-14s %-12s %-14s\n" "iprof" "$T_hot_ip" "$T_proc_ip" "$n_events_ip" "$analysis_ip"
printf "%-12s %-12s %-14s %-12s %-14s\n" "Kineto" "$T_hot_kn" "$T_proc_kn" "$n_events_kn" "$analysis_kn"

echo ""
echo "=== Time Estimate for Full Experiment ==="
echo "(one rep = BL + IP + KN wall times + iprof replay)"
total_rep=$(awk "BEGIN { printf \"%.1f\", $T_proc_bl + $T_proc_ip + $T_proc_kn + $analysis_ip }")
echo "  Time per rep:  ~${total_rep} s"
for reps in 5 10 15; do
    est=$(awk "BEGIN { printf \"%.1f\", $total_rep * $reps / 60 }")
    echo "  $reps reps:      ~${est} min"
done
echo ""
echo "Raw results written to: $RAW"
