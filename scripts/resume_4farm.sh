#!/bin/bash
#SBATCH --job-name=large_4farm
#SBATCH --output=logs/large_4farm.out
#SBATCH --error=logs/large_4farm.err
#SBATCH --partition=gpu-4farm
#SBATCH --gres=gpu:4
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

WORK_DIR="/home/mhjang01/workspace/flex"
PYTHON_BIN="/home/mhjang01/.conda/envs/flex/bin/python"

ulimit -n 100000
/home/mhjang01/.conda/envs/flex/bin/torchrun --standalone --nproc_per_node=4 "$WORK_DIR/train_multigpu.py" --config "$WORK_DIR/configs/large.yaml"

echo "=============================================="
echo "Ending time: $(date)"
echo "=============================================="
