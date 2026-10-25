#!/bin/bash
#SBATCH --job-name=p1_jaxtari_sweep
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=results/slurm/%x-%A_%a.out
#SBATCH --error=results/slurm/%x-%A_%a.err
set -euo pipefail
WT=/cluster/maier/UVCS_p1check
cd $WT
export PYTHONPATH=$WT/jaxtari JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
echo "=== node=$(hostname) task=$SLURM_ARRAY_TASK_ID $(date -u +%FT%TZ) ==="
jaxtari/.venv/bin/python -c "import jaxtari; print(jaxtari.__file__)"
for i in $(seq $SLURM_ARRAY_TASK_ID 2 63); do
  jaxtari/.venv/bin/python results/p1check/sweep_one.py $WT results/p1check/xref results/p1check/sweep $i > results/p1check/log_$i.txt 2>&1 &
done
wait
echo "=== DONE $(date -u +%FT%TZ) ==="
