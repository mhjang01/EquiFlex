# EquiFlex

Flexible protein–ligand docking with an SE(3)-equivariant flow-matching model.
Given a receptor structure, a set of pocket residues, and a ligand SMILES,
EquiFlex jointly samples ligand poses and pocket side-chain conformations.

## Installation

Requires Python 3.12, a CUDA 12.8 GPU, and [`uv`](https://docs.astral.sh/uv/).

```bash
git lfs install                 # model weights are stored via Git LFS
git clone https://github.com/mhjang01/EquiFlex.git
cd EquiFlex
uv sync                         # resolves the locked environment + installs equiflex
```

`uv sync` installs the `equiflex` package (src layout) in editable mode. Prefix
commands with `uv run`, or activate the environment with `source .venv/bin/activate`.

The trained checkpoint (`weights/equiflex_90m.ckpt`, ~358 MB) is tracked with Git
LFS; ensure `git lfs install` ran before cloning or it will appear as a text pointer.

## Layout

```
src/equiflex/
  model/        SE(3)-equivariant flow-matching model, Lightning trainer, loss
  prep/         structure parsing + featurization (ESM3, UniMol, graph features)
  inference/    flow-matching sampler, guidance, relaxation, scoring
  dataprep/     dataset parse + featurize entrypoints, split CSVs
scripts/        train.py, run_inference.py, extract_weights.py
configs/        training configs (base / large / v5_*)
weights/        trained checkpoint (Git LFS)
data/5SAK_ZRY/  bundled example system for inference
```

## Inference

A ready-to-run example (PoseBusters system `5SAK_ZRY`) ships with precomputed
features, so no ESM3/UniMol download is needed:

```bash
uv run python scripts/run_inference.py --config data/5SAK_ZRY/inference.yaml
```

Outputs land in `data/5SAK_ZRY/`: per-sample `sample_*/complex.cif` and
`ligand.sdf`, plus `results.pt` and RMSD stats in `data/stats.jsonl`.

### Your own system

Point the config at a receptor PDB/CIF, the pocket residues (author numbering),
and a ligand SMILES. Delete any stale `system_data/` so features are recomputed:

```yaml
checkpoint: weights/equiflex_90m.ckpt
output_root: ./inference_output
num_samples: 10
num_steps: 50
schedule: uniform
device: cuda
systems:
  my_system:
    protein:
      path: /path/to/receptor.pdb
      residues:
        A: [42, 43, 44, 80]
    ligand:
      smiles: "CC(=O)Nc1ccc(O)cc1"
      reference_sdf: /path/to/ref_ligand.sdf   # optional, enables RMSD
```

Featurizing a new system loads ESM3 and UniMol (downloaded on first use).
Single-system CLI mode is also available:

```bash
uv run python scripts/run_inference.py \
    --checkpoint weights/equiflex_90m.ckpt \
    --receptor protein.pdb \
    --residues A:42 A:43 A:44 \
    --smiles "CC(=O)Nc1ccc(O)cc1" \
    --output-dir ./output
```

## Data preparation

Training/eval data is prepared in two stages. Each source system is a directory
containing `receptor.cif` and `ligand_files/*.sdf`.

**Stage 1 — parse** structures into per-system intermediates:

```python
from pathlib import Path
from equiflex.dataprep.metadata_extraction import parse_single_system

parse_single_system(
    system_dir=Path("raw/5SAK_ZRY"),   # holds receptor.cif + ligand_files/
    storage_dir=Path("data"),          # per-system output dir is created here
)
```

**Stage 2 — featurize** the parsed systems (graph features + ESM3 + UniMol):

```python
from equiflex.dataprep.featurize_data import main as featurize

featurize(data_dir="data", run_esm3=True, run_unimol=True, device="cuda")
```

This writes the tensors the model consumes into each system directory:
`ligand_features.pt`, `pocket_features.pt`, `esm3_embeddings.pt`,
`unimol_embeddings.pt`, and `ligands.pkl`.

Dataset splits (LP-PDBBind, PLINDER) are provided as CSVs under
`src/equiflex/dataprep/lp_pdbbind/` and `lp_plinder/`.

## Training

```bash
uv run python scripts/train.py --config configs/base.yaml --data-dir /path/to/data
```

Configs under `configs/` select model size and training hyperparameters;
`--resume <ckpt>` continues from a checkpoint.

## Extracting weights from a training checkpoint

`scripts/extract_weights.py` produces a slim, inference-only checkpoint: it bakes
the EMA weights into the state dict and drops optimizer/scheduler/loop state.

```bash
uv run python scripts/extract_weights.py train.ckpt weights/equiflex_90m.ckpt
```
