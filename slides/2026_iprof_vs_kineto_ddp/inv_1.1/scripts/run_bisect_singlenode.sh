#!/usr/bin/env bash
#PBS -A Performance
#PBS -q debug
#PBS -l walltime=00:20:00
#PBS -l filesystems=home
#PBS -l select=1
#PBS -o /home/avivasmeza/dev/iprof_vs_kineto/notes/slides/2026_iprof_vs_kineto_ddp/inv_1.1/results/run_bisect_singlenode.out
#PBS -e /home/avivasmeza/dev/iprof_vs_kineto/notes/slides/2026_iprof_vs_kineto_ddp/inv_1.1/results/run_bisect_singlenode.err

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
echo "cp=0  before init_process_group  (expected: ze~0 if bug present)"
echo "cp=1  after  init_process_group"
echo "cp=2  after  DDP.__init__        (expected: ze>0 if bug present)"
echo "cp=3  after  first all_reduce"
echo "cp=4  after  first matmul"
echo ""

for CP in 0 1 2 3 4; do
    $MPIEXEC python3 "$WKLD/repro_bisect_mpi4py.py" --checkpoint "$CP" --steps "$STEPS" || true
done
