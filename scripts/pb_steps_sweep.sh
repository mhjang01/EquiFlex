#!/bin/bash
# PoseBusters redock success vs number of Euler integration steps, guidance on,
# released 90M checkpoint, 10 samples/system, seed 42. One run_inference root
# per step count under data/posebusters/sweep/steps_<N>, featurisation shared
# with data/posebusters/prepared via system_data symlinks (no re-featurising),
# real-Vina rescoring, then scripts/eval_posebusters_redock.py per root.
#
#   STEPS="10 20 30 40 50 100" bash scripts/pb_steps_sweep.sh
#
# Outputs: data/posebusters/sweep/steps_<N>/{stats.jsonl,vina_real.csv,pb_metrics.csv}
# and data/posebusters/sweep/summary.csv via scripts/pb_sweep_summary.py.
set -euo pipefail
RAW="${RAW:-$HOME/workspace/equiflex/comparison/diffbindfr_run/pb_dataset}"
STEPS="${STEPS:-10 20 30 40 50 100}"
SWEEP=data/posebusters/sweep
PREPARED=data/posebusters/prepared
export VINA_BIN="${VINA_BIN:-$HOME/references/dock/vina/bin/vina}"
PY=.venv/bin/python
mkdir -p "$SWEEP"

for n in $STEPS; do
    root="$SWEEP/steps_$n"
    mkdir -p "$root"
    for sd in "$PREPARED"/*/system_data; do
        id=$(basename "$(dirname "$sd")")
        mkdir -p "$root/$id"
        [ -e "$root/$id/system_data" ] || ln -s "../../../prepared/$id/system_data" "$root/$id/system_data"
    done
    $PY scripts/make_posebusters_inference_config.py --raw-root "$RAW" --num-steps "$n" \
        --output-root "$root" --out "$root/inference.yaml"
    if [ ! -f "$root/stats.jsonl" ] || [ "$(wc -l < "$root/stats.jsonl")" -lt 308 ]; then
        echo "=== inference: $n steps  $(date)"
        $PY scripts/run_inference.py --config "$root/inference.yaml" > "$root/inference.log" 2>&1
    fi
    echo "=== vina rescoring: $n steps  $(date)"
    $PY scripts/vina_rescore.py --run "steps_$n=$root" --out "$root/vina_real.csv" > "$root/vina.log" 2>&1
    echo "=== posebusters eval: $n steps  $(date)"
    $PY scripts/eval_posebusters_redock.py --gt-dir "$RAW" --run "steps_$n=$root" \
        --vina-csv "$root/vina_real.csv" --out "$root/pb_metrics.csv" > "$root/eval.log" 2>&1
done

$PY scripts/pb_sweep_summary.py --sweep "$SWEEP" --steps $STEPS
