#!/usr/bin/env python3
"""Prepare the PoseBusters benchmark for EquiFlex inference.

The benchmark manifest uses the same ``systems`` schema as run_inference.py,
but paths are resolved from ``--raw-root`` so it remains portable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from run_inference import _parse_residues_spec, prepare_system, system_data_is_current


def resolve_input(raw_root: Path, system_id: str, configured: str | None, suffix: str) -> Path:
    if configured:
        candidate = Path(configured)
        portable = raw_root / system_id / candidate.name
        if portable.exists():
            return portable
        if candidate.exists():
            return candidate
    return raw_root / system_id / f"{system_id}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text())
    systems = list(manifest["systems"].items())
    if args.limit is not None:
        systems = systems[: args.limit]

    jobs = []
    missing_inputs = []
    for system_id, spec in systems:
        protein = spec["protein"]
        ligand = spec["ligand"]
        receptor = resolve_input(
            args.raw_root, system_id, protein.get("path"), "_protein.pdb"
        )
        reference = resolve_input(
            args.raw_root,
            system_id,
            ligand.get("reference_sdf"),
            "_ligand.sdf",
        )
        if not receptor.exists():
            missing_inputs.append(str(receptor))
            continue
        if not reference.exists():
            missing_inputs.append(str(reference))
            continue
        jobs.append((system_id, protein, ligand, receptor, reference))

    if missing_inputs:
        preview = "\n".join(f"  {p}" for p in missing_inputs[:10])
        raise SystemExit(
            f"Missing {len(missing_inputs)} benchmark input file(s):\n{preview}"
        )

    pending = []
    for job in jobs:
        system_dir = args.output_root / job[0] / "system_data"
        # Same currency test as run_inference.py (files present and feature keys
        # up to date), so "Ready" here means run_inference will not re-featurise.
        if args.force or system_data_is_current(system_dir) is not None:
            pending.append(job)

    print(f"Manifest systems: {len(systems)}")
    print(f"Ready: {len(jobs) - len(pending)}  To prepare: {len(pending)}")
    if args.check_only or not pending:
        return

    import torch
    from equiflex.prep.featurize import ESM3Extractor, UniMolExtractor

    esm = ESM3Extractor(device=args.device)
    unimol = UniMolExtractor(model_size="1.1B")
    failures = []
    for index, (system_id, protein, ligand, receptor, _reference) in enumerate(pending, 1):
        print(f"[{index}/{len(pending)}] {system_id}")
        try:
            prepare_system(
                receptor_path=receptor,
                chain_residues=_parse_residues_spec(protein["residues"]),
                smiles=ligand["smiles"],
                system_dir=args.output_root / system_id / "system_data",
                device=args.device,
                esm_extractor=esm,
                unimol_extractor=unimol,
                ions=ligand.get("ions"),
            )
        except Exception as exc:
            failures.append((system_id, str(exc)))
            print(f"  [FAIL] {exc}")

    del esm, unimol
    torch.cuda.empty_cache()
    print(f"Prepared: {len(pending) - len(failures)}  Failed: {len(failures)}")
    if failures:
        failure_log = args.output_root / "prepare_failures.tsv"
        failure_log.parent.mkdir(parents=True, exist_ok=True)
        failure_log.write_text("".join(f"{sid}\t{msg}\n" for sid, msg in failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
