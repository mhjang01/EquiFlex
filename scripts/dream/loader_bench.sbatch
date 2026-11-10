#!/bin/bash
#SBATCH --job-name=eqf_loader
#SBATCH --account=test
#SBATCH --partition=jobs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/loader_%j.out
#SBATCH --error=logs/loader_%j.err
# Training dataloader throughput alone (no GPU): is I/O the bottleneck?
#   CONFIG=configs/v8_79m.yaml sbatch scripts/dream/loader_bench.sbatch
WORK_DIR="${WORK_DIR:-$SLURM_SUBMIT_DIR}"; CONFIG="${CONFIG:-configs/v8_79m.yaml}"
cd "$WORK_DIR" || exit 1
singularity exec --bind /scratch:/scratch --bind "$WORK_DIR:/work" --pwd /work "$WORK_DIR/equiflex.sif" python - "$CONFIG" <<'PY' 2>&1 | grep -v "Warn\|libcue\|mathbf\|frac"
import sys, time, torch
from equiflex.model import load_config
from equiflex.model.loader import get_train_val_dataloaders
cfg = load_config(sys.argv[1]); d, t = cfg.data, cfg.training
for workers in (16, 8):
    loader, *_ = get_train_val_dataloaders(data_dir=d.data_dir, batch_size=d.batch_size, num_workers=workers, val_ratio=d.val_ratio, seed=t.seed, log_dir=None,
        val_batches=d.val_batches, split_csv=d.split_csv, split_dir=None, preload=False, systems_per_batch=d.systems_per_batch, pad_systems=d.pad_systems)
    it = iter(loader); n = 0; t0 = None
    for b in it:
        if n == 20: t0 = time.time()      # skip warm-up
        n += 1
        if n == 220: break
    dt = time.time() - t0
    print(f'num_workers={workers}: {200/dt:.1f} microbatches/s per rank ({dt/200*1000:.0f} ms each); training needs ~1.7/s per rank at 8 copies')
PY
