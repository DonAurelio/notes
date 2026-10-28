#!/usr/bin/env bash
#PBS -A Performance
#PBS -q debug
#PBS -l walltime=00:10:00
#PBS -l filesystems=home
#PBS -l select=2
#PBS -o /home/avivasmeza/dev/iprof_vs_kineto/notes/slides/2026_iprof_vs_kineto_ddp/inv_1.0/results/run_repro5.out
#PBS -e /home/avivasmeza/dev/iprof_vs_kineto/notes/slides/2026_iprof_vs_kineto_ddp/inv_1.0/results/run_repro5.err

set -eo pipefail

mapfile -t HOSTS < <(sort -u "$PBS_NODEFILE")
HOST0=${HOSTS[0]}
HOST1=${HOSTS[1]}

REPO=/home/avivasmeza/dev/iprof_vs_kineto
INV=$REPO/notes/slides/2026_iprof_vs_kineto_ddp/inv_1.0
WKLD=$INV/workloads
CCL_WRAP=$REPO/notes/slides/2026_iprof_vs_kineto_ddp/workloads/ccl_local_wrap.sh
STEPS=3

source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
module use /home/avivasmeza/thapi_env/modules
module load thapi-env/blob
module load frameworks

export MASTER_ADDR=$HOST0
export PALS_WORLD_SIZE=2
MPIEXEC="mpiexec --no-transfer -n 2 -ppn 1 -hosts $HOST0,$HOST1"


echo "Hosts: $HOST0  $HOST1"
echo "Steps: $STEPS"
echo ""

echo "=== [WIDE] profile opens before dist.init_process_group ==="
$MPIEXEC "$CCL_WRAP" python3 "$WKLD/repro_kineto_ze.py" --scope wide --steps "$STEPS"

echo ""
echo "=== [NARROW] profile opens after dist.init_process_group ==="
$MPIEXEC "$CCL_WRAP" python3 "$WKLD/repro_kineto_ze.py" --scope narrow --steps "$STEPS"
