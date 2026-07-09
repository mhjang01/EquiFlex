"""Reproduce PLINDER's PL50-train versus PoseBusters PLI leakage aggregate.

For each PoseBusters-matched PLINDER system, retain the largest ``pli_qcov``
score to any PL50-train system across both stored score directions. This is
the orientation used by PLINDER's historical ``train_posebusters`` leakage
routine. The primary target set uses exact (PDB ID, ligand CCD) matches;
the optional PDB-only fallback is reported as a sensitivity analysis.
"""

import argparse
import glob
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PL50 = "/mnt/drive/plinder/2024-06/v0/splits/plinder-pl50.parquet"
ANNOTATIONS = "/mnt/drive/new_plinder/v2/index/annotation_table.parquet"
PB_IDS = "/home/mhjang/workspace/flex/prep/posebusters_pdb_ccd_ids.txt"
SCORE_GLOBS = ["/mnt/drive/plinder/2024-06/v2/scores/search_db=holo/*.parquet"]
OUTPUT = Path("/home/mhjang/workspace/flex/data_v2_plinder/logs/pl50_paper_pli_qcov.json")


def make_pb_sets() -> tuple[set[str], set[str], dict[str, object]]:
    pairs = {
        (line.split("_", 1)[0].lower(), line.split("_", 1)[1].upper())
        for line in Path(PB_IDS).read_text().splitlines()
        if line.strip()
    }
    annotations = pq.read_table(
        ANNOTATIONS, columns=["system_id", "entry_pdb_id", "ligand_ccd_code"]
    ).to_pandas()
    annotations["pdb"] = annotations["entry_pdb_id"].str.lower()
    annotations["ccd"] = annotations["ligand_ccd_code"].astype(str).str.upper()
    pb_pdbs = {pdb for pdb, _ in pairs}
    candidates = annotations[annotations["pdb"].isin(pb_pdbs)]
    exact = candidates[
        [(pdb, ccd) in pairs for pdb, ccd in zip(candidates["pdb"], candidates["ccd"])]
    ]
    exact_systems = set(exact["system_id"])
    unmatched_pdbs = pb_pdbs - set(exact["pdb"])
    fallback_systems = set(candidates[candidates["pdb"].isin(unmatched_pdbs)]["system_id"])
    return exact_systems, exact_systems | fallback_systems, {
        "pb_pairs": len(pairs),
        "pb_pdbs": len(pb_pdbs),
        "exact_systems": len(exact_systems),
        "exact_pdbs": exact["pdb"].nunique(),
        "fallback_pdbs": sorted(unmatched_pdbs),
        "fallback_systems": len(fallback_systems),
    }


def best_score_per_pb(train: set[str], pb_systems: set[str], metric: str) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("PRAGMA threads=16")
    con.execute("PRAGMA memory_limit='24GB'")
    con.execute("PRAGMA temp_directory='/tmp/duckdb_spill'")
    con.execute("CREATE TEMP TABLE train_sys (system_id VARCHAR)")
    con.executemany("INSERT INTO train_sys VALUES (?)", [(system,) for system in train])
    con.execute("CREATE TEMP TABLE pb_sys (system_id VARCHAR)")
    con.executemany("INSERT INTO pb_sys VALUES (?)", [(system,) for system in pb_systems])
    shards = sorted(path for pattern in SCORE_GLOBS for path in glob.glob(pattern))
    parts = []
    for direction in ("forward", "reverse"):
        for i, shard in enumerate(shards, 1):
            start = time.monotonic()
            if direction == "forward":
                select = "target_system AS pb_system"
                where = "query_system IN (SELECT system_id FROM train_sys) AND target_system IN (SELECT system_id FROM pb_sys)"
            else:
                select = "query_system AS pb_system"
                where = "query_system IN (SELECT system_id FROM pb_sys) AND target_system IN (SELECT system_id FROM train_sys)"
            print(f"[{direction} {i}/{len(shards)}] {Path(shard).name}", flush=True)
            rows = con.execute(f"SELECT {select}, MAX(similarity) AS max_pli_qcov FROM read_parquet('{shard}') WHERE metric='{metric}' AND {where} GROUP BY pb_system").fetchdf()
            parts.append(rows)
            print(f"  {len(rows)} PB systems in {time.monotonic()-start:.1f}s", flush=True)
    con.close()
    return pd.concat(parts, ignore_index=True).groupby("pb_system", as_index=False)["max_pli_qcov"].max()


def summarize(pb_systems: set[str], scores: pd.DataFrame) -> dict[str, object]:
    score_by_pb = dict(zip(scores["pb_system"], scores["max_pli_qcov"]))
    values = np.array([score_by_pb.get(system, 0.0) for system in sorted(pb_systems)])
    return {
        "n_pb_systems": len(pb_systems),
        "n_pb_systems_with_score": len(scores),
        "fraction_gt_50": float((values > 50).mean()),
        "fraction_ge_50": float((values >= 50).mean()),
        "count_gt_50": int((values > 50).sum()),
        "count_ge_50": int((values >= 50).sum()),
        "mean": float(values.mean()),
        "max": float(values.max()),
        "percentiles": {f"p{q}": float(np.percentile(values, q)) for q in [50, 75, 90, 95, 99]},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label", default="PL50")
    parser.add_argument("--metric", default="pli_qcov")
    args = parser.parse_args()
    split = (
        pd.read_csv(args.split_csv)
        if args.split_csv is not None
        else pq.read_table(PL50, columns=["system_id", "split"]).to_pandas()
    )
    train = set(split.loc[split["split"] == "train", "system_id"])
    exact, fallback, target_info = make_pb_sets()
    print(f"{args.label} train: {len(train)}; exact PB systems: {len(exact)}", flush=True)

    fallback_scores = best_score_per_pb(train, fallback, args.metric)
    summary = {
        "method": f"PLINDER train_posebusters aggregation: max bidirectional {args.metric} per PB system",
        "n_train": len(train),
        "pb_mapping": target_info,
        "exact_pdb_ccd": summarize(
            exact, fallback_scores[fallback_scores["pb_system"].isin(exact)]
        ),
        "pdb_fallback_sensitivity": summarize(fallback, fallback_scores),
    }
    OUTPUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
