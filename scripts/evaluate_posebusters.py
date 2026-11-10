"""Run PoseBusters checks on Flex inference outputs.

The evaluator scans a Flex output root containing one directory per complex and
``sample_*`` subdirectories, reading ``ligand.sdf`` and ``complex.cif``. Poses
are evaluated as sampled: there is no post-hoc relaxation step.

This runs the ligand-only ``dock`` checks (no ground-truth pose needed). For
the redock protocol with RMSD, side-chain RMSD and Vina-ranked selection see
``scripts/eval_posebusters_redock.py``.

Example:
    python scripts/evaluate_posebusters.py \
        --output-root data/posebusters/prepared \
        --limit 10 \
        --out-csv data/posebusters/posebusters_dock_10.csv
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, List, Tuple

import gemmi
import pandas as pd
from posebusters import PoseBusters
from tqdm import tqdm


def iter_prediction_pairs(
    output_root: Path,
    max_samples_per_complex: int,
) -> Iterable[Tuple[str, int, Path, Path, bool]]:
    for complex_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        sample_dirs = sorted(
            (
                p
                for p in complex_dir.iterdir()
                if p.is_dir() and p.name.startswith("sample_")
            ),
            key=lambda p: int(p.name.split("_", 1)[1]),
        )
        for sample_dir in sample_dirs[:max_samples_per_complex]:
            sample_idx = int(sample_dir.name.split("_", 1)[1])
            ligand = sample_dir / "ligand.sdf"
            complex_file = sample_dir / "complex.cif"
            if ligand.exists() and complex_file.exists():
                yield complex_dir.name, sample_idx, ligand, complex_file


def write_pdb_if_needed(complex_file: Path, tmp_dir: Path) -> Path:
    if complex_file.suffix.lower() not in {".cif", ".mmcif"}:
        return complex_file
    pdb_file = tmp_dir / f"{complex_file.parent.parent.name}_{complex_file.parent.name}.pdb"
    st = gemmi.read_structure(str(complex_file))
    st.write_pdb(str(pdb_file))
    return pdb_file


def evaluate(output_root: Path, limit: int, max_samples_per_complex: int) -> pd.DataFrame:
    buster = PoseBusters("dock")
    pairs = list(iter_prediction_pairs(output_root, max_samples_per_complex))
    if limit is not None:
        pairs = pairs[:limit]

    rows: List[pd.DataFrame] = []
    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for complex_id, sample_idx, ligand, complex_file in tqdm(pairs):
            try:
                mol_cond = write_pdb_if_needed(complex_file, tmp_dir)
                df = buster.bust(mol_pred=str(ligand), mol_cond=str(mol_cond))
                df["complex_id"] = complex_id
                df["sample"] = sample_idx
                df["ligand_path"] = str(ligand)
                df["complex_path"] = str(complex_file)
                df["mol_cond_path"] = str(mol_cond)
                df["pb_valid"] = df.drop(
                    columns=[
                        "complex_id",
                        "sample",
                        "ligand_path",
                        "complex_path",
                        "mol_cond_path",
                    ],
                    errors="ignore",
                ).fillna(False).all(axis=1)
                rows.append(df)
            except Exception as exc:
                rows.append(
                    pd.DataFrame(
                        [
                            {
                                "complex_id": complex_id,
                                "sample": sample_idx,
                                "ligand_path": str(ligand),
                                "complex_path": str(complex_file),
                                "pb_valid": False,
                                "error": str(exc),
                            }
                        ]
                    )
                )

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.Series:
    bool_cols = [
        col
        for col in df.columns
        if len(df[col].dropna()) > 0
        and df[col].dropna().map(lambda x: isinstance(x, bool)).all()
    ]
    summary = {
        f"{col}_rate": df[col].fillna(False).astype(bool).mean()
        for col in bool_cols
    }
    summary["n"] = len(df)
    return pd.Series(summary).sort_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True,
                        help="run_inference.py output_root")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-samples-per-complex", type=int, default=1)
    parser.add_argument("--out-csv", type=Path, default=Path("posebusters_eval.csv"))
    parser.add_argument("--summary-csv", type=Path, default=None)
    args = parser.parse_args()

    df = evaluate(
        output_root=args.output_root,
        limit=args.limit,
        max_samples_per_complex=args.max_samples_per_complex,
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    if df.empty:
        print("No predictions found.")
        return

    summary = summarize(df)
    print(summary.to_string())
    if args.summary_csv is not None:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_frame("value").to_csv(args.summary_csv)


if __name__ == "__main__":
    main()
