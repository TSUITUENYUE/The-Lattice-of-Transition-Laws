"""FID of a CIFAR-10 lattice model decoded along one schedule: one cell of Table 7.

    python -m lattice.experiments.read_cifar --ckpt CKPT --model NAME --name CELL --out DIR \
        --schedule synchronised|sequential|sloped [--slope K] [--order raster|spread] \
        [--n 50000] [--chunk 500] [--seed 0] \
        --reference DATA/cifar10/fid_reference --fid-cache DATA/cifar10/fid_cache
    # a long cell sampled by S jobs (K = 0 .. S-1), then scored once
    ... --shards S --shard K --sample-only
    ... --shards S --score-only

The schedules are the paths of sampling/pixel_paths.py. On 16 patches and 128 transitions the
synchronised path takes 128 calls, the sloped path 128 + 15 K (143, 188 and 368 at slopes 1, 4
and 16) and the sequential path 2063, 15 of them calls that move nothing. The spread order visits
the patches of a sloped path in the bit-reversed order of the patch grid in place of raster order;
a raster cell and a spread cell draw the same labels and the same noise, so they are paired, and
they are different cells: every row names its order.

The protocol of every cell: --n class-balanced labels (n/10 per class, the classes in order), the
stochastic coupling, the model's EMA weights, chunks of --chunk samples, and one CPU generator per
shard seeded with seed * 1000 + K. Shard K owns the slice [n K / S, n (K + 1) / S) of the sample
index, so the labels do not depend on S and the noise does: the paper sampled every sequential
cell as four shards and every other cell as one. FID against the 50,000 training images
(evaluation/fid_cifar.py).

A shard keeps its samples and its generator's state after every chunk, so an interrupted read
resumes where it stopped and draws the noise an uninterrupted read would have drawn. Scoring
writes OUT/CELL.csv, one row (model, schedule, slope, order, calls, samples, seed, shards, fid;
calls is the path's, whichever process scores), and deletes the samples.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import torch

from ..core.levels import edm_levels
from ..data.cifar10 import class_balanced, onehot
from ..evaluation.fid_cifar import fid, write_png
from ..models import pixel as PX
from ..sampling.pixel_paths import SCHEDULES, path_calls, spread_order, walk
from ..utils.results import Rows


def sample_shard(args, kind, slope, sdir, dev):
    """Sample shard --shard of the cell into ``sdir``, resuming from its saved progress."""
    model, ck = PX.load(args.ckpt, dev)
    levels = edm_levels(ck["transitions"])
    patches, n_patches = PX.patch_index(32, ck["patch"])
    patches = patches.to(dev)
    order = spread_order(n_patches) if args.order == "spread" else None
    lo = args.n * args.shard // args.shards
    hi = args.n * (args.shard + 1) // args.shards
    labels = class_balanced(args.n, dev)[lo:hi]
    g = torch.Generator().manual_seed(args.seed * 1000 + args.shard)
    prog = sdir / f"progress_{args.shard}.pt"
    key = dict(ckpt=str(Path(args.ckpt).resolve()), schedule=kind, slope=slope, order=args.order,
               n=args.n, chunk=args.chunk, seed=args.seed, shards=args.shards, shard=args.shard)
    done = 0
    state = torch.load(prog, weights_only=False) if prog.exists() else None
    if state is not None and state["key"] == key:
        g.set_state(state["generator"])
        done = state["done"]
        print(f"  {args.name} shard {args.shard}: resumed at {done}/{hi - lo}", flush=True)
    t0 = time.time()
    while done < hi - lo:
        oh = onehot(labels[done:done + args.chunk])
        x, calls = walk(model, oh, kind, levels, patches, n_patches, slope=slope, generator=g,
                        device=dev, order=order)
        assert calls == path_calls(kind, n_patches, ck["transitions"], slope)
        write_png(x, sdir, start=lo + done)
        done += len(oh)
        tmp = prog.with_suffix(".tmp")
        torch.save({"key": key, "generator": g.get_state(), "done": done}, tmp)
        os.replace(tmp, prog)
        print(f"  {args.name} shard {args.shard}: {done}/{hi - lo} ({time.time() - t0:.0f}s)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="the model's checkpoint (train_cifar)")
    ap.add_argument("--model", required=True, help="the model's name in the row")
    ap.add_argument("--name", required=True, help="the cell's name: OUT/NAME.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--schedule", required=True, choices=SCHEDULES)
    ap.add_argument("--slope", type=int, default=0, help="the slope of the sloped path")
    ap.add_argument("--order", default="raster", choices=("raster", "spread"),
                    help="the order a sequential or sloped path visits the patches in")
    ap.add_argument("--n", type=int, default=50000, help="samples in the cell (a multiple of 10)")
    ap.add_argument("--chunk", type=int, default=500, help="samples a batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1, help="jobs that share the sampling")
    ap.add_argument("--shard", type=int, default=0, help="the shard this job samples")
    ap.add_argument("--sample-only", action="store_true", help="sample this shard and stop")
    ap.add_argument("--score-only", action="store_true", help="score the samples every shard wrote")
    ap.add_argument("--reference", default="", help="directory of the 50,000 reference PNGs")
    ap.add_argument("--fid-cache", default="", help="where the reference features are cached")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    kind = args.schedule
    if kind == "sloped" and args.slope < 1:
        ap.error("the sloped path needs --slope of at least 1")
    slope = args.slope if kind == "sloped" else 0
    if args.n % 10:
        ap.error("--n must be a multiple of 10 (class-balanced labels)")
    if not args.sample_only and not (args.reference and args.fid_cache):
        ap.error("scoring needs --reference and --fid-cache")
    dev = args.device
    out = Path(args.out)
    sdir = out / f".samples_{args.name}"
    sdir.mkdir(parents=True, exist_ok=True)
    if not args.score_only:
        sample_shard(args, kind, slope, sdir, dev)
    if args.sample_only:
        print(f"SHARD_DONE {args.name} {args.shard}/{args.shards}", flush=True)
        return
    missing = [i for i in range(args.n) if not (sdir / f"{i:06d}.png").exists()]
    if missing:
        raise SystemExit(f"{sdir} lacks {len(missing)} of {args.n} samples (first {missing[0]:06d}); "
                         "score once every shard has finished")
    extra = len(list(sdir.glob("*.png"))) - args.n
    if extra:
        raise SystemExit(f"{sdir} holds {extra} samples beyond the first {args.n}")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    n_patches = PX.patch_index(32, ck["patch"])[1]
    value = fid(sdir, args.reference, args.fid_cache, cuda=dev.startswith("cuda"))
    with Rows(args.name, out) as rows:
        rows.add(model=args.model, schedule=kind, slope=slope, order=args.order,
                 calls=path_calls(kind, n_patches, ck["transitions"], slope),
                 samples=args.n, seed=args.seed, shards=args.shards, fid=value)
    shutil.rmtree(sdir, ignore_errors=True)


if __name__ == "__main__":
    main()
