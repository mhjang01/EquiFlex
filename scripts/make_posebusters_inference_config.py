"""Create a Flex batch-inference YAML for prepared PoseBusters systems.

This is intended for the 308 prepared directories under
``/home/mhjang/workspace/flex/test_output_abl``.  The generated YAML reuses each
directory's ``system_data`` and therefore does not require re-featurization.

Example:
    python make_posebusters_inference_config.py \
        --systems-root /home/mhjang/workspace/flex/test_output_abl \
        --out /home/mhjang/workspace/flex/outputs/posebusters_308_relax.yaml

Then run:
    python run_inference.py --config /home/mhjang/workspace/flex/outputs/posebusters_308_relax.yaml
"""

import argparse
from pathlib import Path

import yaml


def prepared_system_dirs(systems_root: Path):
    for system_dir in sorted(p for p in systems_root.iterdir() if p.is_dir()):
        if (system_dir / "system_data" / "ligand_features.pt").exists():
            yield system_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--systems-root",
        type=Path,
        default=Path("/home/mhjang/workspace/flex/test_output_abl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/mhjang/workspace/flex/outputs/posebusters_308_relax.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/mhjang/workspace/flex/test_output_abl"),
        help="Where run_inference.py should write per-system outputs.",
    )
    parser.add_argument(
        "--checkpoint",
        default="/home/mhjang/workspace/flex/checkpoints/b200/last.ckpt",
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--schedule", choices=["linear", "log"], default="log")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-relax", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    systems = {}
    for system_dir in prepared_system_dirs(args.systems_root):
        if args.limit is not None and len(systems) >= args.limit:
            break

        sample_complex = system_dir / "sample_0" / "complex.cif"
        systems[system_dir.name] = {
            "protein": {
                # Used only for full-protein writing because system_data exists.
                # Do not run this config with force_prepare=true.
                "path": str(sample_complex if sample_complex.exists() else system_dir),
                "residues": {"A": [1]},
            },
            "ligand": {"smiles": "C"},
            "output_dir": str(args.output_root / system_dir.name),
        }

    config = {
        "checkpoint": args.checkpoint,
        "num_samples": args.num_samples,
        "num_steps": args.num_steps,
        "schedule": args.schedule,
        "seed": args.seed,
        "device": args.device,
        "relax": not args.no_relax,
        "force_prepare": False,
        "output_root": str(args.output_root),
        "systems": systems,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"Wrote {len(systems)} systems to {args.out}")


if __name__ == "__main__":
    main()
