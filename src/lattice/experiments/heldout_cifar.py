"""The held-out objective of trained CIFAR-10 lattice models, at fixed draws.

    python -m lattice.experiments.heldout_cifar --ckpt A.pt [B.pt ...] --measure mixed \
        --data-root DATA/cifar10 --out DIR [--name heldout_cifar] [--draws 4]

Each model's own training objective (EDM's weighting under its own training distribution) on
CIFAR-10's 10,000 test images, every image read at --draws states, EMA weights. The levels come
from a CPU generator and the noise from a generator on --device, both seeded afresh for every
checkpoint, so two checkpoints of one model differ only through their weights; it says whether a
model stopped at a step count was still improving on its objective (the paper's mixed model: at
60,000 and at 120,000 steps). The noise is drawn on the device, so two readings compare only on
one kind of card. Writes OUT/NAME.csv, one row per checkpoint.
"""
from __future__ import annotations

import argparse
import math

import torch

from ..core.levels import edm_levels
from ..data.cifar10 import SDATA, Cifar
from ..models import pixel as PX
from ..training.measures import MEASURES, level_drawer
from ..utils.results import Rows


@torch.no_grad()
def heldout_objective(net, test, measure, transitions, patches, n_patches, knots,
                      draws=4, chunk=500, seed=20260910):
    """Mean weighted denoising error of ``net`` on ``test`` under ``measure``."""
    dev = test.x.device
    gl = torch.Generator().manual_seed(seed)
    gn = torch.Generator(device=dev).manual_seed(seed)
    draw = level_drawer(measure, transitions, n_patches, gl, dev)
    was = net.training
    net.eval()
    tot, n = 0.0, 0
    for _ in range(draws):
        for i in range(0, len(test), chunk):
            x0 = test.x[i:i + chunk]
            oh = test.eye[test.y[i:i + chunk]]
            smap = PX.level_map(draw(x0.shape[0]), patches, knots)
            yn = x0 + smap * torch.randn(x0.shape, generator=gn, device=dev)
            lam = (smap ** 2 + SDATA ** 2) / (smap * SDATA) ** 2
            err = lam * (net.denoise(yn, smap, oh) - x0) ** 2
            tot += float(err.mean(dim=(1, 2, 3)).sum())
            n += x0.shape[0]
    net.train(was)
    return tot / n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True, help="checkpoints of one model (train_cifar)")
    ap.add_argument("--measure", required=True, choices=MEASURES, help="the models' training distribution")
    ap.add_argument("--draws", type=int, default=4, help="states read per test image")
    ap.add_argument("--limit", type=int, default=0, help="the first this many test images only")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="heldout_cifar", help="OUT/NAME.csv")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    dev = args.device
    test = Cifar(args.data_root, device=dev, train=False)
    if args.limit:
        test.x, test.y = test.x[:args.limit], test.y[:args.limit]
    with Rows(args.name, args.out) as rows:
        for path in args.ckpt:
            net, ck = PX.load(path, dev)
            if ck["measure"] != args.measure:
                raise SystemExit(f"{path} was trained on {ck['measure']}, not {args.measure}")
            levels = edm_levels(ck["transitions"])
            knots = torch.tensor([math.sqrt(float(k)) for k in levels.knots], device=dev)
            patches, n_patches = PX.patch_index(32, ck["patch"])
            v = heldout_objective(net, test, args.measure, ck["transitions"], patches.to(dev),
                                  n_patches, knots, draws=args.draws)
            rows.add(ckpt=path, model=ck["tag"], step=ck["step"], measure=args.measure,
                     draws=args.draws, images=len(test), heldout_objective=round(v, 6))


if __name__ == "__main__":
    main()
