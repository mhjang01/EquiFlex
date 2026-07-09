"""Extract a distributable, inference-only checkpoint from a training checkpoint.

Training checkpoints carry optimizer state, LR schedulers, loop/callback state and
a separate EMA shadow — none needed for inference. This bakes the EMA weights into
the model state_dict and keeps only what ``FlexTrainer.load_from_checkpoint`` needs.

    python scripts/extract_weights.py <train.ckpt> <out.ckpt>
"""

import argparse
import dataclasses
import importlib
from pathlib import Path

import torch

# Training checkpoints were written with the flat-layout package names
# (``model``, ``prep``, ``inference``). Rewrite pickled config dataclasses to the
# ``equiflex.*`` namespace so the distributed checkpoint loads without them.
NAMESPACE_REMAP = {
    "model": "equiflex.model",
    "prep": "equiflex.prep",
    "inference": "equiflex.inference",
}


def _remap_class(module: str) -> str:
    for old, new in NAMESPACE_REMAP.items():
        if module == old or module.startswith(old + "."):
            return new + module[len(old):]
    return module


def remap_namespaces(obj):
    """Recursively rebuild dataclass instances under the equiflex namespace."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        new_module = _remap_class(type(obj).__module__)
        cls = getattr(importlib.import_module(new_module), type(obj).__name__)
        kwargs = {f.name: remap_namespaces(getattr(obj, f.name))
                  for f in dataclasses.fields(obj)}
        return cls(**kwargs)
    if isinstance(obj, dict):
        return {k: remap_namespaces(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(remap_namespaces(v) for v in obj)
    return obj


def extract(src: Path, dst: Path) -> None:
    ck = torch.load(src, map_location="cpu", weights_only=False)
    state_dict = dict(ck["state_dict"])

    ema = ck.get("ema_state_dict")
    if ema is not None:
        shadow = ema["shadow"]  # model-relative keys, no "model." prefix
        applied = 0
        for name, tensor in shadow.items():
            key = f"model.{name}"
            if key in state_dict:
                state_dict[key] = tensor
                applied += 1
        print(f"Baked EMA weights into state_dict (decay={ema['decay']}, {applied} tensors)")

    slim = {
        "state_dict": state_dict,
        "hyper_parameters": remap_namespaces(ck["hyper_parameters"]),
        "hparams_name": ck["hparams_name"],
        "pytorch-lightning_version": ck["pytorch-lightning_version"],
        "epoch": ck.get("epoch"),
        "global_step": ck.get("global_step"),
    }

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, dst)
    src_mb = src.stat().st_size / 1e6
    dst_mb = dst.stat().st_size / 1e6
    print(f"{src} ({src_mb:.0f} MB) -> {dst} ({dst_mb:.0f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", type=Path, help="Training checkpoint (.ckpt)")
    p.add_argument("dst", type=Path, help="Output inference checkpoint (.ckpt)")
    args = p.parse_args()
    extract(args.src, args.dst)


if __name__ == "__main__":
    main()
