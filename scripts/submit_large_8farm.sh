#!/bin/bash
#SBATCH --job-name=large_8farm
#SBATCH --output=logs/large_8farm.out
#SBATCH --error=logs/large_8farm.err
#SBATCH --partition=gpu-8farm
#SBATCH --gres=gpu:8
#SBATCH --account=gpu
#SBATCH --cpus-per-task=28  
#SBATCH --mem=224G         
#SBATCH --time=1-00:00:00

mkdir -p logs

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Starting time: $(date)"
echo "=============================================="

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCHRUN="${TORCHRUN:-$WORK_DIR/.venv/bin/torchrun}"

ulimit -n 100000
cd "$WORK_DIR" && "$TORCHRUN" --standalone --nproc_per_node=8 scripts/train.py --config configs/v10_512.yaml "$@"

echo "=============================================="
echo "Ending time: $(date)"
echo "=============================================="
