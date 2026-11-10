#!/usr/bin/env bash
# Rebuild a flat dir of symlinks at the remote training host.
#
# The loader (model/loader.py: FlexDataset) resolves each system as
# data_dir/<system_id> (FLAT, no split subdir) and reads the train/val ID
# lists from split_train.txt / split_val.txt in split_dir. This script builds
# that flat surface from the split-subdir tree produced by the prep pipeline.
#
# Reads pre-computed split lists (NOT the stale splits.csv):
#   data_v2_plinder/split_train.txt   (one system_id per line)
#   data_v2_plinder/split_val.txt
#
# Output:
#   data_v2_plinder_flat/<system_id> -> ../data_v2_plinder/<split>/<system_id>
#   data_v2_plinder_flat/split_train.txt   (copied; use as split_dir)
#   data_v2_plinder_flat/split_val.txt
#
# Test is NOT flattened: consume it via test_data_dir=data_v2_plinder/test.
#
# Usage (run on the remote, from the parent of data_v2_plinder/):
#   bash data_v2_plinder/flatten_remote.sh
#
# Train with: data_dir=<...>/data_v2_plinder_flat  split_dir=<...>/data_v2_plinder_flat
# Test  with: test_data_dir=<...>/data_v2_plinder/test

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
SRC="$ROOT_DIR/data_v2_plinder"
DST="$ROOT_DIR/data_v2_plinder_flat"

for f in split_train.txt split_val.txt; do
    if [[ ! -f "$SRC/$f" ]]; then
        echo "ERROR: $SRC/$f missing. Run from the parent of data_v2_plinder/." >&2
        exit 1
    fi
done

mkdir -p "$DST"
linked=0; skipped=0; missing=0

link_split() {
    local split="$1" listfile="$2"
    while IFS= read -r sid; do
        [[ -z "$sid" ]] && continue
        local src="$SRC/$split/$sid"
        local dst="$DST/$sid"
        if [[ -L "$dst" || -e "$dst" ]]; then skipped=$((skipped+1)); continue; fi
        if [[ ! -d "$src" ]]; then missing=$((missing+1)); continue; fi
        ln -s "../data_v2_plinder/$split/$sid" "$dst"
        linked=$((linked+1))
    done < <(tr -d '\r' < "$SRC/$listfile")
}

link_split train split_train.txt
link_split val   split_val.txt

cp "$SRC/split_train.txt" "$DST/split_train.txt"
cp "$SRC/split_val.txt"   "$DST/split_val.txt"

echo "linked=$linked skipped=$skipped missing=$missing"
echo "split_train.txt: $(wc -l < "$DST/split_train.txt")"
echo "split_val.txt:   $(wc -l < "$DST/split_val.txt")"
echo "flat dir: $DST"
echo "train: data_dir=$DST split_dir=$DST"
echo "test:  test_data_dir=$SRC/test"
