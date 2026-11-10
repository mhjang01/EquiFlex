#!/bin/bash
#SBATCH --job-name=eqf_memprof
#SBATCH --account=test
#SBATCH --partition=jobs
#SBATCH --nodes=1
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/memprof_%j.out
#SBATCH --error=logs/memprof_%j.err
# scripts/dream/mem_profile.py on one A100-40GB inside the image.
#   CONFIG=configs/v6_60m.yaml sbatch scripts/dream/mem_profile.sbatch
WORK_DIR="${WORK_DIR:-$SLURM_SUBMIT_DIR}"
CONFIG="${CONFIG:-configs/v6_60m.yaml}"
PICK="${PICK:-p50 p90 p99 max}"
DATA_DIR="${DATA_DIR:-data/smoke/train}"
TIMESTEPS="${TIMESTEPS:-4}"   # copies per system; copies > TIMESTEPS become extra systems
COPIES="${COPIES:-1 2 4 8 12 16 24 32}"
unset SLURM_NTASKS SLURM_NPROCS SLURM_STEP_NUM_TASKS
cd "$WORK_DIR" || exit 1
singularity exec --nv --bind /scratch:/scratch --bind "$WORK_DIR:/work" --pwd /work "$WORK_DIR/equiflex.sif" \
    python scripts/dream/mem_profile.py --config "$CONFIG" --data-dir "$DATA_DIR" --pick $PICK --copies $COPIES --timesteps $TIMESTEPS 2>&1 | grep -v Warning
