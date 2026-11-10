#!/bin/bash
#SBATCH --job-name=equiflex
#SBATCH --account=test
#SBATCH --partition=jobs
#SBATCH --nodes=1
#SBATCH --gres=gpu:tesla:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=240G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/equiflex_%j.out
#SBATCH --error=logs/equiflex_%j.err
#SBATCH --signal=B:USR1@90
#SBATCH --requeue

# EquiFlex training on dream: one node, four A100-40GB, torchrun inside the
# singularity image built from scripts/dream/equiflex.def.
#
#   mkdir -p logs
#   sbatch scripts/dream/train.sbatch                               # smoke test
#   CONFIG=configs/v5_90m.yaml sbatch scripts/dream/train.sbatch    # full run
#   sbatch scripts/dream/train.sbatch --resume checkpoints/<run>/last.ckpt
#
# Four GPUs is the association limit (GrpTRES gres/gpu=4), so --gres above is
# the maximum. torchrun spawns the four ranks itself, as in
# scripts/submit_4farm.sh, so no --ntasks: with SLURM_NTASKS in the environment
# Lightning's SLURMEnvironment plugin overrides torchrun's rank/device mapping
# and every rank lands on GPU 0. Slurm opens the log files before the script
# runs, so logs/ must already exist. Submit from the repo root: slurm executes
# a spooled copy of this file, so the repo is located via SLURM_SUBMIT_DIR.
#
# The repo is bound at /work and the image imports equiflex from /work/src, so
# code changes need only an rsync. wandb: the compute nodes have outbound
# network, so online logging works; set WANDB_MODE=offline to keep it local.

WORK_DIR="${WORK_DIR:-$SLURM_SUBMIT_DIR}"
SIF_PATH="${SIF_PATH:-$WORK_DIR/equiflex.sif}"
CONFIG="${CONFIG:-configs/dream_smoke.yaml}"
NPROC="${NPROC:-4}"

echo "=============================================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "GPUs     : $CUDA_VISIBLE_DEVICES ($NPROC ranks)"
echo "Image    : $SIF_PATH"
echo "Repo     : $WORK_DIR"
echo "Config   : $CONFIG"
echo "Start    : $(date)"
echo "=============================================="

if [ ! -f "$SIF_PATH" ]; then
    echo "[ERROR] image not found: $SIF_PATH"; exit 1
fi

ulimit -n "$(ulimit -Hn)"
# See the note above: torchrun, not slurm, manages the ranks.
unset SLURM_NTASKS SLURM_NPROCS SLURM_STEP_NUM_TASKS
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / NPROC ))
export WANDB_DIR="$WORK_DIR/logs"
# Expandable segments cut allocator fragmentation: the 134M/nv32 model sits at 35-37 GB of 40.96 on the busiest systems.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$WORK_DIR" || exit 1
singularity exec --nv \
    --bind /scratch:/scratch \
    --bind "$WORK_DIR:/work" \
    --pwd /work \
    "$SIF_PATH" \
    torchrun --standalone --nproc_per_node="$NPROC" scripts/train.py --config "$CONFIG" "$@"
rc=$?

echo "=============================================="
echo "End      : $(date)"
echo "Exit     : $rc"
echo "=============================================="
exit $rc
