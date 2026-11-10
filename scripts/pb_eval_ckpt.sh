#!/bin/bash
# PoseBusters redock evaluation of one checkpoint with the benchmark recipe
# (10 samples, 20 Euler steps, guidance on, seed 42), comparable with the
# released-90M numbers in data/posebusters/sweep. Full training checkpoints
# work as-is: run_inference.py applies the EMA shadow when present.
#
#   bash scripts/pb_eval_ckpt.sh checkpoints/v7_134m/step=12000-val/*.ckpt v7_step12000
#   STEPS=30 bash scripts/pb_eval_ckpt.sh <ckpt> <name>
#
# Outputs under data/posebusters/v7/<name>/: inference.yaml, per-system
# sample dirs, stats.jsonl, vina_real.csv, pb_metrics.csv, eval.log.
# Featurisation is shared with data/posebusters/prepared via symlinks.
set -euo pipefail
CKPT="$1"; NAME="$2"
STEPS="${STEPS:-20}"
RAW="${RAW:-$HOME/workspace/equiflex/comparison/diffbindfr_run/pb_dataset}"
ROOT=data/posebusters/v7/$NAME
PREPARED="${PREPARED:-data/posebusters/prepared}"   # dir of <id>/system_data; the 5 A set is data/posebusters/last_ckpt_20steps_5A_20261015
export VINA_BIN="${VINA_BIN:-$HOME/references/dock/vina/bin/vina}"
PY=.venv/bin/python
[ -f "$CKPT" ] || { echo "no such checkpoint: $CKPT" >&2; exit 1; }
mkdir -p "$ROOT"
for sd in "$PREPARED"/*/system_data; do
    id=$(basename "$(dirname "$sd")")
    mkdir -p "$ROOT/$id"
    [ -e "$ROOT/$id/system_data" ] || ln -s "$(realpath "$sd")" "$ROOT/$id/system_data"
done
$PY scripts/make_posebusters_inference_config.py --raw-root "$RAW" --num-steps "$STEPS" \
    --checkpoint "$CKPT" --output-root "$ROOT" --out "$ROOT/inference.yaml" ${EXTRA:-}
if [ ! -f "$ROOT/stats.jsonl" ] || [ "$(wc -l < "$ROOT/stats.jsonl")" -lt 308 ]; then
    echo "=== inference: $NAME, $STEPS steps  $(date)"
    $PY scripts/run_inference.py --config "$ROOT/inference.yaml" > "$ROOT/inference.log" 2>&1
fi
echo "=== vina rescoring  $(date)"
$PY scripts/vina_rescore.py --run "$NAME=$ROOT" --out "$ROOT/vina_real.csv" > "$ROOT/vina.log" 2>&1
SCORES=(--vina-csv "$ROOT/vina_real.csv")
SMINA_BIN="${SMINA_BIN:-$HOME/references/dock/smina/smina.static}"
if [ -x "$SMINA_BIN" ]; then
    echo "=== vinardo rescoring (smina)  $(date)"
    $PY scripts/smina_rescore.py --run "$NAME=$ROOT" --out "$ROOT/vinardo.csv" --smina "$SMINA_BIN" > "$ROOT/vinardo.log" 2>&1
    SCORES+=(--score-csv "$ROOT/vinardo.csv")
fi
echo "=== posebusters eval  $(date)"
$PY scripts/eval_posebusters_redock.py --gt-dir "$RAW" --run "$NAME=$ROOT" \
    "${SCORES[@]}" --out "$ROOT/pb_metrics.csv" > "$ROOT/eval.log" 2>&1
echo "=== done  $(date)"
grep -A8 "selector" "$ROOT/eval.log" | tail -9
