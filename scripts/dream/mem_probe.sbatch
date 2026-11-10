#!/bin/bash
#SBATCH --job-name=eqf_memprobe
#SBATCH --account=test
#SBATCH --partition=jobs
#SBATCH --nodes=1
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/memprobe_%j.out
#SBATCH --error=logs/memprobe_%j.err

# Peak GPU memory of a training step on one A100-40GB as a function of copies
# per microbatch (data.batch_size x data.systems_per_batch), at num_unrolls=1
# and no gradient checkpointing. Each setting runs train.py for a few steps on
# the smoke subset while nvidia-smi samples memory every second; "CUDA OOM" in
# the output marks a setting that does not fit.
#
#   mkdir -p logs; sbatch scripts/dream/mem_probe.sbatch
#   SETTINGS="4x1 4x2 4x4" GRAD_CKPT=true sbatch scripts/dream/mem_probe.sbatch

WORK_DIR="${WORK_DIR:-$SLURM_SUBMIT_DIR}"
SIF_PATH="${SIF_PATH:-$WORK_DIR/equiflex.sif}"
BASE="${BASE:-configs/dream_smoke.yaml}"
SETTINGS="${SETTINGS:-4x1 4x2 4x3 4x4 4x6 4x8}"   # batch_size x systems_per_batch
GRAD_CKPT="${GRAD_CKPT:-false}"
STEPS="${STEPS:-12}"
unset SLURM_NTASKS SLURM_NPROCS SLURM_STEP_NUM_TASKS
cd "$WORK_DIR" || exit 1
PROBE=logs/memprobe_$SLURM_JOB_ID; mkdir -p "$PROBE"
echo "node $SLURMD_NODENAME gpu $CUDA_VISIBLE_DEVICES base $BASE grad_ckpt $GRAD_CKPT unroll 1 steps $STEPS"
printf '%-8s %-7s %-7s %-10s %s\n' setting copies peak_MB status note

for s in $SETTINGS; do
    b=${s%x*}; n=${s#*x}; cfg=$PROBE/$s.yaml
    sed -e "s|^  batch_size: .*|  batch_size: $b|" \
        -e "s|^  systems_per_batch: .*|  systems_per_batch: $n|" \
        -e "s|^  num_unrolls: .*|  num_unrolls: 1|" \
        -e "s|^  unroll_curriculum: .*|  unroll_curriculum: false|" \
        -e "s|^  use_gradient_checkpointing: .*|  use_gradient_checkpointing: $GRAD_CKPT|" \
        -e "s|^  max_steps: .*|  max_steps: $STEPS|" \
        -e "s|^  accumulate_grad_batches: .*|  accumulate_grad_batches: 1|" \
        -e "s|^  val_check_interval: .*|  val_check_interval: 100000|" \
        -e "s|^  checkpoint_every_n_steps: .*|  checkpoint_every_n_steps: 100000|" \
        -e "s|^  checkpoint_dir: .*|  checkpoint_dir: '$PROBE/ckpt'|" \
        -e "s|^  num_workers: .*|  num_workers: 4|" \
        "$BASE" > "$cfg"
    rm -rf "$PROBE/ckpt"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "$PROBE/$s.mem" &
    smi=$!
    singularity exec --nv --bind /scratch:/scratch --bind "$WORK_DIR:/work" --pwd /work \
        "$SIF_PATH" python scripts/train.py --config "$cfg" > "$PROBE/$s.log" 2>&1
    rc=$?; kill $smi
    peak=$(sort -n "$PROBE/$s.mem" | tail -1)
    oom=$(tr '\r' '\n' < "$PROBE/$s.log" | grep -c 'CUDA OOM')
    status=fits; [ "$oom" -gt 0 ] && status="OOM(x$oom)"; [ $rc -ne 0 ] && status="exit$rc"
    note=$(tr '\r' '\n' < "$PROBE/$s.log" | grep -a 'Epoch 0' | tail -1 | grep -o 'train/total_loss=[0-9.]*')
    printf '%-8s %-7s %-10s %-10s %s\n' "$s" $((b*n)) "${peak:-?}" "$status" "$note"
done
