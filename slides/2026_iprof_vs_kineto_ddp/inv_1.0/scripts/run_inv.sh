#!/usr/bin/env bash
#PBS -A Performance
#PBS -q debug
#PBS -l walltime=00:30:00
#PBS -l filesystems=home
#PBS -l select=2
#PBS -o /home/avivasmeza/dev/iprof_vs_kineto/notes/slides/2026_iprof_vs_kineto_ddp/inv_1.0/results/run_inv.out
#PBS -e /home/avivasmeza/dev/iprof_vs_kineto/notes/slides/2026_iprof_vs_kineto_ddp/inv_1.0/results/run_inv.err

set -eo pipefail

mapfile -t HOSTS < <(sort -u "$PBS_NODEFILE")
HOST0=${HOSTS[0]}
HOST1=${HOSTS[1]}

REPO=/home/avivasmeza/dev/iprof_vs_kineto
INV=$REPO/notes/slides/2026_iprof_vs_kineto_ddp/inv_1.0
WKLD=$INV/workloads
RESULTS=$INV/results
CCL_WRAP=$REPO/notes/slides/2026_iprof_vs_kineto_ddp/workloads/ccl_local_wrap.sh

source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
module use /home/avivasmeza/thapi_env/modules
module load thapi-env/blob
module load frameworks

export MASTER_ADDR=$HOST0
export PALS_WORLD_SIZE=2

STEPS=5
MODEL_ARGS="--device xpu --steps $STEPS --batch-size 1 --seq-len 16 --dim 64 --layers 1 --heads 4 --vocab-size 128 --bf16"
MPIEXEC="mpiexec --no-transfer -n 2 -ppn 1 -hosts $HOST0,$HOST1"

OF=$(mktemp)
trap 'rm -f "$OF"' EXIT

echo "Hosts: $HOST0  $HOST1"
echo "Steps: $STEPS"
echo ""

echo "=== [NARROW] profile wraps step loop only ==="
$MPIEXEC "$CCL_WRAP" \
    python3 "$WKLD/train_llama3_kineto_narrow.py" $MODEL_ARGS > "$OF" 2>>"$RESULTS/run_inv.err"
grep "n_events=" "$OF"
sed -n '/=== key_averages breakdown ===/,/=== end breakdown ===/p' "$OF" > "$RESULTS/narrow_breakdown.txt"
echo "  Written: narrow_breakdown.txt ($(wc -l < "$RESULTS/narrow_breakdown.txt") lines)"

echo ""
echo "=== [WIDE] profile wraps from after select_device ==="
$MPIEXEC "$CCL_WRAP" \
    python3 "$WKLD/train_llama3_kineto_wide.py" $MODEL_ARGS > "$OF" 2>>"$RESULTS/run_inv.err"
grep "n_events=" "$OF"
sed -n '/=== key_averages breakdown ===/,/=== end breakdown ===/p' "$OF" > "$RESULTS/wide_breakdown.txt"
echo "  Written: wide_breakdown.txt ($(wc -l < "$RESULTS/wide_breakdown.txt") lines)"

echo ""
echo "=== Diff analysis ==="
python3 "$INV/scripts/diff_breakdown.py"
