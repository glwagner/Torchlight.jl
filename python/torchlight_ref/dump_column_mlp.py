"""Generate Torchlight fixtures for the column MLP.

    .venv/bin/python -m torchlight_ref.dump_column_mlp --out data/fixtures

Writes, for each (size, dtype):
    column_mlp_<size>_<dtype>.h5          eval mode, derivatives, SGD/Adam steps
    column_mlp_<size>_<dtype>_train.h5    train mode with explicit dropout masks
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from .column_mlp import PAPER_WIDTHS, TINY_WIDTHS, ColumnMLP
from .export import export_column_mlp_fixture, file_sha256

SIZES = {"tiny": (TINY_WIDTHS, 13), "full": (PAPER_WIDTHS, 64)}
DTYPES = {"float64": torch.float64, "float32": torch.float32}


def build(size: str, dtype: torch.dtype, seed: int):
    widths, batch = SIZES[size]
    torch.manual_seed(seed)
    model = ColumnMLP(widths, dropout_p=0.1).to(dtype)
    gen = torch.Generator().manual_seed(seed)
    # Inputs at O(1) scale; the paper standardises physical predictors.
    x = torch.randn(batch, widths[0], dtype=dtype, generator=gen)
    target = torch.randn(batch, widths[-1], dtype=dtype, generator=gen)
    masks = [(torch.rand(batch, w, dtype=dtype, generator=gen) >= model.dropout_p).to(dtype)
             for w in widths[1:-1]]
    return model, x, target, masks, gen


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fixtures")
    ap.add_argument("--sizes", nargs="+", default=list(SIZES), choices=list(SIZES))
    ap.add_argument("--dtypes", nargs="+", default=list(DTYPES), choices=list(DTYPES))
    ap.add_argument("--seed", type=int, default=20260910)
    args = ap.parse_args(argv)

    torch.use_deterministic_algorithms(True)
    manifest = {}
    for size in args.sizes:
        for dname in args.dtypes:
            dtype = DTYPES[dname]
            model, x, target, masks, gen = build(size, dtype, args.seed)
            model_id = f"column_mlp_{size}"
            base = os.path.join(args.out, f"{model_id}_{dname}")
            p = export_column_mlp_fixture(base + ".h5", model=model, x=x, target=target,
                                          mode="eval", model_id=model_id, seed=args.seed, gen=gen)
            manifest[os.path.basename(p)] = file_sha256(p)
            p = export_column_mlp_fixture(base + "_train.h5", model=model, x=x, target=target,
                                          mode="train", model_id=model_id, seed=args.seed,
                                          masks=masks, gen=gen)
            manifest[os.path.basename(p)] = file_sha256(p)
            print("wrote", base + "{,_train}.h5", "params",
                  sum(p.numel() for p in model.parameters()))
    with open(os.path.join(args.out, "MANIFEST.json"), "w") as fh:
        json.dump({"torch": torch.__version__, "seed": args.seed, "sha256": manifest}, fh, indent=2)


if __name__ == "__main__":
    main()
