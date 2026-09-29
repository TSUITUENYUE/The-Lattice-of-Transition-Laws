"""Samples and intermediate states of a CIFAR-10 lattice model along its schedules, saved as arrays.

    # cells: the first --n samples at a fixed seed at five schedules (the image cells of Figure 2)
    python -m lattice.experiments.sample_grid cells --ckpt CKPT --out DIR --name NAME [--n 4] [--seed 0]
    # states: one sample walked along five schedules, its state at --moments moments (Figure 6)
    python -m lattice.experiments.sample_grid states --ckpt CKPT --out DIR [--label 7] [--seed 7]

cells walks the synchronised path, the sloped path in raster order at slopes 1, 4 and 16, and the
sequential path, with labels 0, 1, ..., n - 1 and a CPU generator seeded afresh for every schedule,
so no cell can be chosen by hand. Writes OUT/NAME.npz: for each schedule (synchronised, sloped1,
sloped4, sloped16, sequential) the samples as uint8 (n, 32, 32, 3), and the labels and call counts.

states walks the synchronised path, the sloped path in raster order at slopes 8 and 32, the sloped
path at slope 32 in spread order, and the sequential path, each from the same seed on a CPU
generator with one label (7, horse), and records the walk at --moments evenly spaced path indices
t = round(i T / (moments - 1)) of its T steps. Every recorded state is the walk's own: pixels at
different levels at the same moment. Writes OUT/graded_states.npz:
  schedule, slope, order    the five rows
  calls                     model calls of each row's path (128, 248, 608, 608, 2063)
  moment                    (5, moments) the path indices recorded
  state                     (5, moments, 3, 32, 32) the state y at each of them, float32 (the
                            starting noise first; the last is the state after the final call)
  level                     (5, moments, 32, 32) the level index each pixel sits at in that
                            state, 0 at the noisy end and the number of transitions at the clean end
  sample                    (5, 3, 32, 32) the walk's output, its last estimate of the clean image
  transitions, patch, label, seed
The display figure06 (experiments/tables_images.py) reads the level maps.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..core.levels import edm_levels
from ..data.cifar10 import CLASSES, class_balanced, onehot
from ..models import pixel as PX
from ..sampling.pixel_paths import path_calls, spread_order, walk

CELL_SCHEDULES = (("synchronised", 0), ("sloped", 1), ("sloped", 4), ("sloped", 16), ("sequential", 0))
# (schedule, slope, order)
STATE_SCHEDULES = (("synchronised", 0, "raster"), ("sloped", 8, "raster"), ("sloped", 32, "raster"),
                   ("sloped", 32, "spread"), ("sequential", 0, "raster"))


def _open(args):
    model, ck = PX.load(args.ckpt, args.device)
    patches, n_patches = PX.patch_index(32, ck["patch"])
    return model, ck, edm_levels(ck["transitions"]), patches.to(args.device), n_patches


def cells(args):
    if not 1 <= args.n <= CLASSES:
        raise SystemExit("--n takes 1 to 10 samples, one per class")
    model, ck, levels, patches, n_patches = _open(args)
    labels = class_balanced(CLASSES, args.device)[:args.n]
    arrays = {"labels": labels.cpu().numpy()}
    calls = []
    for kind, slope in CELL_SCHEDULES:
        g = torch.Generator().manual_seed(args.seed)
        x, c = walk(model, onehot(labels), kind, levels, patches, n_patches, slope=slope,
                    generator=g, device=args.device)
        arrays[f"{kind}{slope or ''}"] = ((x.clamp(-1, 1) + 1) * 127.5).round().byte() \
            .permute(0, 2, 3, 1).cpu().numpy()
        calls.append(c)
        print(f"CELL {args.name} {kind}{slope or ''}: {c} calls", flush=True)
    arrays["calls"] = np.array(calls)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{args.name}.npz", **arrays)
    print(f"WROTE {out / f'{args.name}.npz'}", flush=True)


def states(args):
    model, ck, levels, patches, n_patches = _open(args)
    oh = onehot(torch.tensor([args.label], device=args.device))
    moment, state, level, sample, calls = [], [], [], [], []
    for kind, slope, order_name in STATE_SCHEDULES:
        order = spread_order(n_patches) if order_name == "spread" else None
        steps = path_calls(kind, n_patches, len(levels), slope)
        snaps = sorted({int(round(i * steps / (args.moments - 1))) for i in range(args.moments)})
        ys, lvs = [], []

        def record(t, y, lv, snaps=snaps, ys=ys, lvs=lvs):
            if t in snaps:
                ys.append(y[0].float().cpu().numpy())
                lvs.append(lv[patches].cpu().numpy().astype(np.int16))

        g = torch.Generator().manual_seed(args.seed)
        x, c = walk(model, oh, kind, levels, patches, n_patches, slope=slope, generator=g,
                    device=args.device, order=order, record=record)
        print(f"STATES {kind} slope={slope} order={order_name}: {c} calls, moments {snaps}", flush=True)
        moment.append(snaps)
        state.append(np.stack(ys))
        level.append(np.stack(lvs))
        sample.append(x[0].float().cpu().numpy())
        calls.append(c)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "graded_states.npz",
             schedule=np.array([s[0] for s in STATE_SCHEDULES]),
             slope=np.array([s[1] for s in STATE_SCHEDULES]),
             order=np.array([s[2] for s in STATE_SCHEDULES]),
             calls=np.array(calls), moment=np.array(moment), state=np.stack(state),
             level=np.stack(level), sample=np.stack(sample),
             transitions=len(levels), patch=ck["patch"], label=args.label, seed=args.seed)
    print(f"WROTE {out / 'graded_states.npz'}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["cells", "states"])
    ap.add_argument("--ckpt", required=True, help="the model's checkpoint (train_cifar)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="cells", help="cells: OUT/NAME.npz")
    ap.add_argument("--n", type=int, default=4, help="cells: samples a cell, labels 0 .. n-1")
    ap.add_argument("--seed", type=int, default=None, help="default 0 for cells, 7 for states")
    ap.add_argument("--label", type=int, default=7, help="states: the class (7 is horse)")
    ap.add_argument("--moments", type=int, default=6, help="states: moments recorded a path")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if args.seed is None:
        args.seed = 0 if args.command == "cells" else 7
    (cells if args.command == "cells" else states)(args)


if __name__ == "__main__":
    main()
