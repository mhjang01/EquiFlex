#!/bin/bash
#SBATCH --job-name=arch_sweep
#SBATCH --account=test
#SBATCH --partition=jobs
#SBATCH --nodes=1
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err

# One-GPU architecture sweep on dream (param count, training-step VRAM and time).
#   CONFIG=configs/v10_512.yaml GRID="use_gradient_checkpointing=False gradient_checkpoint_every=2" sbatch scripts/dream/sweep.sbatch
# Submit from the repo root; the repo is located via SLURM_SUBMIT_DIR as in train.sbatch.

WORK_DIR="${WORK_DIR:-$SLURM_SUBMIT_DIR}"
SIF_PATH="${SIF_PATH:-$WORK_DIR/equiflex.sif}"
CONFIG="${CONFIG:-configs/v10_512.yaml}"
GRID="${GRID:-}"
N_REC="${N_REC:-450}"
N_LIG="${N_LIG:-50}"

echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME, GPU $CUDA_VISIBLE_DEVICES, config $CONFIG, grid [$GRID], $(date)"
unset SLURM_NTASKS SLURM_NPROCS SLURM_STEP_NUM_TASKS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$WORK_DIR" || exit 1
# shellcheck disable=SC2086  # CONFIG and GRID are intentionally word-split (several configs / grid sets)
singularity exec --nv \
    --bind /scratch:/scratch \
    --bind "$WORK_DIR:/work" \
    --pwd /work \
    "$SIF_PATH" \
    python scripts/arch_sweep.py --configs $CONFIG ${GRID:+--grid $GRID} --vram --n-rec "$N_REC" --n-lig "$N_LIG"
rc=$?
echo "Exit $rc, $(date)"
exit $rc
