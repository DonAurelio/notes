#!/usr/bin/env bash
#PBS -A Performance
#PBS -q debug
#PBS -l walltime=00:10:00
#PBS -l filesystems=home
#PBS -l select=1
#PBS -o /home/avivasmeza/dev/iprof_vs_kineto/notes/slides/2026_iprof_vs_kineto_ddp/inv_1.1/results/run_repro_final.out
#PBS -e /home/avivasmeza/dev/iprof_vs_kineto/notes/slides/2026_iprof_vs_kineto_ddp/inv_1.1/results/run_repro_final.err

set -eo pipefail

HOST0=$(sort -u "$PBS_NODEFILE" | head -1)

REPO=/home/avivasmeza/dev/iprof_vs_kineto
WKLD=$REPO/notes/slides/2026_iprof_vs_kineto_ddp/inv_1.1/workloads
STEPS=3

source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
module use /home/avivasmeza/thapi_env/modules
module load thapi-env/blob
module load frameworks

export MASTER_ADDR=$HOST0
export PALS_WORLD_SIZE=2
MPIEXEC="mpiexec --no-transfer -n 2 -ppn 2 -hosts $HOST0"

echo "Host: $HOST0 (single node, ppn=2)"
echo "Steps: $STEPS"
echo ""

echo "=== [WIDE] profile before DDP init — expected ze_events ~ 0 ==="
$MPIEXEC python3 "$WKLD/repro_kineto_ddp.py" --scope wide --steps "$STEPS"

echo ""
echo "=== [NARROW] profile after DDP init — expected ze_events > 0 ==="
$MPIEXEC python3 "$WKLD/repro_kineto_ddp.py" --scope narrow --steps "$STEPS"
