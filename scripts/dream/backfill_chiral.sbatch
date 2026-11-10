#!/bin/bash
#SBATCH --job-name=eqf_chiral
#SBATCH --account=test
#SBATCH --partition=jobs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/chiral_%j.out
#SBATCH --error=logs/chiral_%j.err
# scripts/backfill_chiral_indices.py over the featurised dataset (CPU only).
#   DATA="data_v2_plinder/train data_v2_plinder/val data_v2_plinder/test" sbatch scripts/dream/backfill_chiral.sbatch
WORK_DIR="${WORK_DIR:-$SLURM_SUBMIT_DIR}"
DATA="${DATA:-data_v2_plinder/train data_v2_plinder/val data_v2_plinder/test}"
cd "$WORK_DIR" || exit 1
singularity exec --bind /scratch:/scratch --bind "$WORK_DIR:/work" --pwd /work "$WORK_DIR/equiflex.sif" \
    python scripts/backfill_chiral_indices.py $DATA 2>&1 | grep -v Warning
