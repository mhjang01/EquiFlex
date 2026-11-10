#!/bin/bash
#SBATCH --job-name=equiflex
#SBATCH --account=lcdd
#SBATCH --partition=6000ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:6000ada:4
#SBATCH --qos=long
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/equiflex_%j.out
#SBATCH --error=logs/equiflex_%j.err
#SBATCH --signal=B:USR1@90
#SBATCH --requeue

# EquiFlex training on mas2: one node, four RTX 6000 Ada (48 GB), native .venv
# (no singularity image here, unlike scripts/dream/train.sbatch).
#
#   mkdir -p logs
#   CONFIG=configs/v10_512_nophys.yaml sbatch scripts/mas2/train.sbatch
#   CONFIG=configs/v10_512_nophys.yaml sbatch scripts/mas2/train.sbatch --resume checkpoints/v10_512_nophys/last.ckpt
#
# mas2's submit filter rejects -c/--cpus-per-task and --mem on GPU partitions:
# it assigns 6 CPUs and 25 GB per GPU automatically, so 4 GPUs gets 24 CPUs and
# 100 GB. That is why configs/v10_512_nophys.yaml sets num_workers 5 (5 x 4 ranks
# + 4 main processes = 24) rather than dream's 16. Association limits (account
# lcdd): GrpTRES gres/gpu=16 across all jobs, MaxTRES gres/gpu=8,node=1 per job.
# Walltime is capped by QOS, not by the partition's 30 days. Options, with their
# multifactor QOS priority (PriorityWeightQOS=10000) and per-user job cap:
#   veryshort 4 h / 10000 / 10   short 12 h / 5000 / 8
#   long      3 d / 2000  / 4    verylong 7 d / 1000 / 2
# A 50k-step run on 6000ada (~1.6x dream's A100 step time) needs ~10 days, so no
# single job covers it. long is the pick: 4 jobs x 3 days = 12 days, double
# verylong's priority factor, and a 3-day request backfills into gaps far more
# easily than a 7-day one against bf_window=10080.
#
# Chain all four as IDENTICAL jobs -- scripts/train.py auto-resumes from the
# newest checkpoints/<run_name>/last*.ckpt when --resume is absent, so no job
# needs to know it is a continuation:
#
#   P=$(CONFIG=configs/v10_512_nophys.yaml sbatch --parsable scripts/mas2/train.sbatch)
#   for i in 2 3 4; do
#     P=$(CONFIG=configs/v10_512_nophys.yaml sbatch --parsable \
#           --dependency=afterany:$P scripts/mas2/train.sbatch)
#   done
#
# This is the gap that killed dream's 229323: SLURMEnvironment is disabled below
# (see the torchrun note), so Lightning does NOT auto-requeue on the USR1 signal
# and a walltime kill is final unless the next job is already queued.
#
# torchrun spawns the four ranks itself, so no --ntasks: with SLURM_NTASKS set,
# Lightning's SLURMEnvironment overrides torchrun's rank/device mapping and
# every rank lands on GPU 0. Slurm opens the log files before the script runs,
# so logs/ must exist. Submit from the repo root ($SLURM_SUBMIT_DIR).

WORK_DIR="${WORK_DIR:-$SLURM_SUBMIT_DIR}"
CONFIG="${CONFIG:-configs/v10_512_nophys.yaml}"
NPROC="${NPROC:-4}"
TORCHRUN="${TORCHRUN:-$WORK_DIR/.venv/bin/torchrun}"

echo "=============================================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "GPUs     : $CUDA_VISIBLE_DEVICES ($NPROC ranks)"
echo "Repo     : $WORK_DIR"
echo "Config   : $CONFIG"
echo "Start    : $(date)"
echo "=============================================="

ulimit -n "$(ulimit -Hn)"
unset SLURM_NTASKS SLURM_NPROCS SLURM_STEP_NUM_TASKS
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_ON_NODE:-24} / NPROC ))
# Four 6000 Ada on gpu3/gpu4 have no NVLink: nvidia-smi topo -m reports GPU0-GPU1
# PIX and GPU2-GPU3 NODE, but both cross pairs SYS (over the UPI link between
# NUMA nodes). With NCCL's defaults -- P2P inside each pair, host transport
# across NUMA -- the first 153 MB DDP parameter broadcast deadlocks: that is what
# hung job 115772 (ranks 1-3 "watchdog got stuck for 480 seconds", SIGABRT
# before step 1). Forcing a single transport for all six pairs fixes it; probed
# with nccl_probe.py, defaults hang while P2P-off, SHM-off and both-off all pass.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$WORK_DIR" || exit 1
"$TORCHRUN" --standalone --nproc_per_node="$NPROC" scripts/train.py --config "$CONFIG" "$@"
rc=$?

echo "=============================================="
echo "End      : $(date)"
echo "Exit     : $rc"
echo "=============================================="
exit $rc
