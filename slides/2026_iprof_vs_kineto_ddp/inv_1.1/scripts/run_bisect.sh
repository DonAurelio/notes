#!/usr/bin/env bash
#PBS -A Performance
#PBS -q debug
#PBS -l walltime=00:20:00
#PBS -l filesystems=home
#PBS -l select=2
#PBS -o /home/avivasmeza/dev/iprof_vs_kineto/notes/slides/2026_iprof_vs_kineto_ddp/inv_1.1/results/run_bisect.out
#PBS -e /home/avivasmeza/dev/iprof_vs_kineto/notes/slides/2026_iprof_vs_kineto_ddp/inv_1.1/results/run_bisect.err

set -eo pipefail

mapfile -t HOSTS < <(sort -u "$PBS_NODEFILE")
HOST0=${HOSTS[0]}
HOST1=${HOSTS[1]}

REPO=/home/avivasmeza/dev/iprof_vs_kineto
INV=$REPO/notes/slides/2026_iprof_vs_kineto_ddp/inv_1.1
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
echo "Checkpoint meaning:"
echo "  cp=0  before init_process_group  (expected: ze=0,  known broken)"
echo "  cp=1  after  init_process_group"
echo "  cp=2  after  DDP.__init__"
echo "  cp=3  after  first all_reduce"
echo "  cp=4  after  first matmul       (expected: ze>0,  known working)"
echo ""

for CP in 0 1 2 3 4; do
    $MPIEXEC "$CCL_WRAP" python3 "$WKLD/repro_bisect.py" --checkpoint "$CP" --steps "$STEPS"
done
