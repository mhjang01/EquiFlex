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

The released checkpoint (`weights/weights.ckpt`, ~306 MB) is tracked with Git LFS;
ensure `git lfs install` ran before cloning or it will appear as a text pointer.

Pose scoring defaults to **GenScore**, which is not vendored here. Point
`$GENSCORE_ROOT` at an upstream [GenScore](https://github.com/sc8668/RTMScore)
checkout containing `GenScore/` and `trained_models/<model>.pth` (default:
`~/references/dock/GenScore`, model `GT_0.0_1`). Without it, set
`scoring.genscore.enabled: false` and select on Vina or Vinardo instead.

Optional, for classical pose ranking: [AutoDock Vina](https://vina.scripps.edu/)
or smina, and [Open Babel](https://openbabel.org/) on `$PATH` (or `$VINA_BIN` /
`$OBABEL_BIN`).

## Layout

```
src/equiflex/
  model/        SE(3)-equivariant flow-matching model, Lightning trainer, loss
  prep/         structure parsing, pocket selection, featurization (ESM3, UniMol)
  inference/    flow-matching sampler, physics guidance, GenScore + Vina scoring
  dataprep/     dataset parse + featurize entrypoints
scripts/
  run_inference.py   single-system CLI or batch YAML inference
  run_hts.py         screen a SMILES library against one pocket
  select_pocket.py   pocket residues at the training cutoff
  train.py           training entrypoint
  patch_unimol.sh    Uni-Mol v2 padding-invariance patch
configs/equiflex_config.yaml      training config of the released checkpoint
weights/weights.ckpt              released checkpoint (Git LFS)
data/5SAK_ZRY/                    bundled example system (inputs + precomputed features)
data/posebusters/benchmark.yaml   308-system PoseBusters manifest (pockets, SMILES, ions)
data_v2_plinder/splits_v9.csv     PLINDER train/val split of the released model
```

## Pocket definition

The model was trained on pockets built by `equiflex.prep.parse.get_pocket_residues`:
every residue with a heavy atom within **5.0 Å** of any ligand heavy atom, minus
residues that collide with the ligand (< 0.1 Å, e.g. a metal duplicated in both
records). Inference pockets must use the same rule; a 4 Å pocket noticeably
degrades results. Given a reference ligand pose:

```bash
uv run python scripts/select_pocket.py \
    --receptor data/5SAK_ZRY/inputs/5SAK_ZRY_protein.pdb \
    --ligand   data/5SAK_ZRY/inputs/5SAK_ZRY_ligand.sdf
```

prints a ready-to-paste `residues:` block (and `ions:` for free metal ions within
the cutoff). `--residues-cli` prints the `--residues A:35 A:37 …` form instead.

Residue numbers are what `parse_structure` assigns, **not** author numbering:
`label_seq_id` for mmCIF input, and for PDB input the 1-based index among the
chain's standard amino-acid residues in file order (modified residues such as
MSE are skipped, so it is not a plain position count either). Always take the
numbers from `select_pocket.py` rather than counting by hand. Every YAML
shipped here records its pocket rule under `pocket_provenance`.

## Inference

A ready-to-run example (PoseBusters system `5SAK_ZRY`) ships with precomputed
features, so no ESM3/UniMol download is needed:

```bash
uv run python scripts/run_inference.py --config data/5SAK_ZRY/inference.yaml
```

Per system, the output directory receives:

| file | content |
|---|---|
| `system_data/` | featurized inputs (reused on the next run unless `force_prepare: true`) |
| `sample_<i>/complex.cif` | sampled pocket side chains merged into the full receptor |
| `sample_<i>/ligand.sdf` | sampled ligand pose (bond orders from the SMILES) |
| `sample_<i>/trajectory_complex.cif` | integration trajectory, with `return_trajectory: true` |
| `results.pt` | raw coordinate tensors, physics losses, ligand RMSD when a reference is given |
| `summary.md` | per-sample physics loss and torch-Vina affinity, with `score_summary: true` |

`<output_root>/stats.jsonl` gets one JSON line per system: atom counts, timing,
per-sample ligand RMSD against `reference_sdf`, or `{"system": ..., "error": ...}`
when featurization or sampling failed (skipped systems are recorded too, so a
batch run is fully accounted for).

### Sampling recipe

The shipped configs carry the recipe used for the reported PoseBusters, PoseX
and PhiBench runs: 10 samples, 20 Euler steps on a uniform schedule, and Tweedie
physics guidance with the block below (the CrossDock run used the same guidance
with 40 samples on the `log` schedule):

```yaml
guidance:
  enabled: true
  t_start: 0.0
  scale: 100.0
  schedule: linear_decay
  bond_coef: 1.0
  angle_coef: 1.0
  vdw_coef: 5.0
  vdw_active_coef: 1.0
  vdw_max_coef: 0.0
  torsion_coef: 0.25
  chiral_coef: 1.0
  chiral_buffer: 0.5236
  grad_clip: 10.0
sde:
  enabled: false
```

The flat-bottom buffers and vdW cutoff of the guidance energy are synced from the
checkpoint's training loss unless set explicitly, so the guidance matches what
the model was trained against. The same values are the defaults of the
single-system CLI (`--guidance` turns it on).

### Your own system

Point a YAML at a receptor PDB/CIF, the pocket residues from `select_pocket.py`,
and a ligand SMILES. Delete any stale `system_data/` (or set `force_prepare: true`)
so features are recomputed:

```yaml
checkpoint: weights/weights.ckpt
output_root: ./inference_output
num_samples: 10
num_steps: 20
schedule: uniform
device: cuda
guidance: {enabled: true, t_start: 0.0, scale: 100.0, schedule: linear_decay,
           bond_coef: 1.0, angle_coef: 1.0, vdw_coef: 5.0, vdw_active_coef: 1.0,
           vdw_max_coef: 0.0, torsion_coef: 0.25, chiral_coef: 1.0,
           chiral_buffer: 0.5236, grad_clip: 10.0}
systems:
  my_system:
    protein:
      path: /path/to/receptor.pdb
      residues:
        A: [42, 43, 44, 80]
    ligand:
      smiles: "CC(=O)Nc1ccc(O)cc1"
      reference_sdf: /path/to/ref_ligand.sdf   # optional, enables RMSD
      ions: ["[Zn+2]"]                         # optional free metal ions
```

Featurizing a new system loads ESM3 and UniMol (downloaded on first use) and
fetches amino-acid reference templates into `./.refs/` in the working directory.
Single-system CLI mode is also available:

```bash
uv run python scripts/run_inference.py \
    --receptor protein.pdb \
    --residues A:42 A:43 A:44 \
    --smiles "CC(=O)Nc1ccc(O)cc1" \
    --guidance \
    --output-dir ./output
```

## Pose scoring

Every scorer sees the receptor the model produced, rebuilt from each pose's own
`complex.cif`, so the sampled side chains are what gets scored. Scoring a
predicted ligand against the input or crystal pocket instead inflates learned
scores substantially.

GenScore is the default selector (see Installation for `$GENSCORE_ROOT`).
`summary.md` also reports a torch port of the Vina function, which is convenient
but does not match the binary pose-for-pose (hydrogen placement and nitrogen
typing differ); `equiflex.inference.vina_binary_score_pose` calls the real
binary in `--score_only` mode on the same rebuilt receptor.

## High-throughput screening

`run_hts.py` screens a SMILES library against a single pocket. The receptor is
parsed and featurized once, ligands are conformer-generated and UniMol-embedded
in batches, and each ligand is sampled against the same pocket while finished
poses are scored in a background pool. Every stage resumes from disk.

```bash
uv run python scripts/run_hts.py --config hts.yaml [--limit N] [--force-prepare]
```

The module docstring documents every YAML key. Outputs are `scores.csv` (one row
per pose), `ranking.csv` (one row per ligand, best score first) and `errors.tsv`;
re-running with the same `output_root` skips prepared shards and scored ligands.

## PoseBusters benchmark

`data/posebusters/benchmark.yaml` lists the 308 systems with 5 Å pockets,
ligand SMILES, free metal ions within the cutoff, and reference SDF names, all
relative to a raw benchmark root laid out as `<id>/<id>_protein.pdb` and
`<id>/<id>_ligand.sdf`. It converts directly into a `run_inference.py` batch
config; the evaluation harness behind the reported numbers is not part of this
tree.

`run_inference.py` reuses `system_data/` if its files and feature keys are
present. It does not check which pocket cutoff produced them, so after changing
a pocket definition re-run with `force_prepare: true` (or `--force-prepare`).

## Data preparation

Training/eval data is prepared in two stages. Each source system is a directory
containing `receptor.cif` and `ligand_files/*.sdf`.

**Stage 1 — parse** structures into per-system intermediates (pocket at 5 Å),
writing `ligands.pkl`, `pocket_residues.pkl` and `sequence_data.pkl` per system:

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
`ligand_features.pt`, `pocket_features.pt`, `esm3_embeddings.pt` and
`unimol_embeddings.pt`.

New ligand features include an ordered atom/entity signature. Uni-Mol extraction
writes the matching signature to `unimol_embeddings.meta.json`; keep this sidecar
with the tensor when copying a prepared system. An explicit preparation run
rebuilds graph features with missing/stale signatures and regenerates unverified
or inconsistent embedding caches, even with `skip_existing=True`. This can
recompute older caches once rather than silently reusing them.

Training and inference reject mismatched ligand/embedding atom counts before the
model forward pass. Legacy datasets without signatures remain readable when
their dimensions agree, but their atom-order provenance is unverified. If ligand
entities or atom order change (including adding/removing ions), rerun preparation;
do not fix the mismatch by blindly truncating or padding embeddings. Evaluation
failures log the split, system context and exception traceback.

The PLINDER split used for the released model is `data_v2_plinder/splits_v9.csv`.

## Training

```bash
uv run python scripts/train.py --config configs/equiflex_config.yaml --data-dir /path/to/data
```

`configs/equiflex_config.yaml` is the configuration the released checkpoint was trained
with. `--data-dir` overrides `data.data_dir`; `data.split_csv` points at the
shipped PLINDER split CSV and `data.test_data_dir` at the held-out systems used
by the RMSD callback, so edit those two in the YAML for your own layout.
`--resume <train.ckpt>` continues from a full training checkpoint; the slim
released weights carry no optimizer state.
