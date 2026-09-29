"""FID of an ImageNet-256 SiT decoded at one corner of the lattice: the cells of Table 8.

    # the samples, in S shards (K = 0 .. S-1), one job each
    python -m lattice.experiments.read_imagenet sample --ckpt CKPT --sample-dir DIR \
        --schedule synchronised|sequential [--calls 16] --shard K --num-shards S [--n 50000]
    # then ADM's evaluation suite on the CPU: FID, sFID, Inception Score, precision, recall
    python -m lattice.experiments.read_imagenet score --sample-dir DIR --adm-dir DATA/adm_evaluator \
        --out OUT --name CELL [--n 50000]

The model regresses the noise of the linear interpolant x_t = t x_1 + (1 - t) x_0, so at level t
its estimate of the clean latent is x_hat = (x_t - (1 - t) eps_hat) / t. In the frame y = x_t / t
the state is the clean latent plus noise at sigma = (1 - t) / t, and every step is the stochastic
coupling of the transition in that frame: from noise variance a to b, with r = b / a,
y_b = r y + (1 - r) x_hat + sqrt(b (a - b) / a) z. The final step returns x_hat.

  synchronised   every patch at one level: the levels k / 250 at k = 1, 2, ..., 249 and then the
                 data end, 249 calls; with --calls C the walk takes C calls on evenly spaced
                 levels from 1 (--calls 16 moves every coordinate as often as the sequential
                 corner does, the last row of Table 8)
  sequential     the 256 patches in raster order, in --groups groups of 16 (one row of the token
                 grid each), one group at a time carried from level 1 to the data end through
                 the levels j / 16, j = 1 .. 16, the rest held: 256 calls. Patches not yet
                 visited wait at level 1 / 250 and finished ones hold at the data end. The
                 mixed training distribution puts its sequential quarter on a front of one
                 patch, and the levels j / 16 and the data end are not among the trained
                 levels k / 250, k = 1 .. 249.

The protocol of every cell: --n samples with labels drawn uniformly over the 1,000 classes from a
CPU generator seeded with --seed, no guidance, the model's EMA weights, and each sample's starting
noise and every step's noise from a CPU generator seeded by the sample's index, so a read does not
depend on the card, the batch or the sharding, and the two models of the table are paired at each
corner. Latents are decoded by sd-vae-ft-ema (divided by 0.18215) to 8-bit PNGs. A shard that is
interrupted resumes at its first batch without all of its samples. Scoring packs the samples,
runs guided-diffusion's evaluator against VIRTUAL_imagenet256_labeled.npz (evaluation/
fid_imagenet.py) and writes OUT/CELL.csv, one row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..data.imagenet_latents import SCALE, load_vae
from ..evaluation.fid_imagenet import adm_score
from ..models import sit
from ..models import sit_levels as SL
from ..utils.results import Rows


def _autocast(x):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda)


@torch.no_grad()
def walk(model, z, y, knots, noise):
    """The synchronised corner: every patch through the levels ``knots`` (increasing in t, the
    last one 1), one call per level."""
    x = z
    for i in range(len(knots) - 1):
        t, tn = knots[i], knots[i + 1]
        tt = torch.full((x.shape[0],), t, device=x.device)
        with _autocast(x):
            eps = model(x, tt, y).float()
        xh = (x - (1.0 - t) * eps) / t
        sa = (1.0 - t) / t
        sb = (1.0 - tn) / tn if tn < 1.0 else 0.0
        a, b = sa * sa, sb * sb
        if b <= 0.0:
            x = tn * xh
            continue
        r = b / a
        tau = b * (a - b) / a
        yv = x / t
        yv = r * yv + (1.0 - r) * xh + (tau ** 0.5) * noise(yv.shape)
        x = tn * yv
    return x


@torch.no_grad()
def walk_sequential(model, z, y, groups, group_levels, levels, noise):
    """The sequential corner: the patches in raster order, one group at a time carried from level
    1 / levels to the data end through ``group_levels`` evenly spaced levels, the rest held."""
    n = z.shape[0]
    side = int(model.x_embedder.num_patches ** 0.5)
    patch = model.x_embedder.patch_size[0]
    t_p = torch.full((n, side * side), 1.0 / levels, device=z.device)
    x = z
    order = torch.arange(side * side, device=z.device)
    per = (side * side + groups - 1) // groups
    for g in range(groups):
        sel = order[g * per:(g + 1) * per]
        for j in range(group_levels):
            tb = (j + 1) / group_levels
            with _autocast(x):
                eps = SL.forward_perpatch(model, x, t_p, y).float()
            te = SL.levels_to_pixels(t_p, side, patch)
            xh = (x - (1.0 - te) * eps) / te
            nxt = t_p.clone()
            nxt[:, sel] = tb
            tn = SL.levels_to_pixels(nxt, side, patch)
            a = ((1.0 - te) / te) ** 2
            b = ((1.0 - tn) / tn).clamp(min=0) ** 2
            # a patch already at the data end (te = 1) has a = b = 0 and keeps its value
            held = a <= 0
            a1 = torch.where(held, torch.ones_like(a), a)
            r = torch.where(held, torch.ones_like(a), (b / a1).clamp(max=1.0))
            tau = torch.where(held, torch.zeros_like(a), (b * (a - b) / a1).clamp(min=0))
            yv = x / te
            yv = r * yv + (1.0 - r) * xh + tau.sqrt() * noise(yv.shape)
            x = tn * yv
            t_p = nxt
    return x


def synchronised_knots(levels, calls):
    """The levels t of the synchronised corner: ``calls`` evenly spaced indices from 1, then 1."""
    idx = [round(1 + i * (levels - 1) / calls) for i in range(calls)] + [levels]
    return [k / levels for k in idx]


def sample(args):
    from PIL import Image
    dev = args.device
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = sit.SiT_models[ck["arch"]](input_size=32, num_classes=1000).to(dev)
    model.load_state_dict(ck["ema"])
    model.eval()
    levels = ck["levels"]
    if args.schedule == "sequential":
        calls = args.groups * args.group_levels
    else:
        knots = synchronised_knots(levels, args.calls or levels - 1)
        calls = len(knots) - 1
    sdir = Path(args.sample_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    meta = dict(ckpt=str(Path(args.ckpt).resolve()), measure=ck["measure"], step=ck["step"],
                schedule=args.schedule, calls=calls, n=args.n, seed=args.seed, num_shards=args.num_shards)
    record = sdir / f"shard_{args.shard}.json"
    if record.exists() and json.loads(record.read_text()) != meta:
        raise SystemExit(f"{sdir} holds samples of another read ({record.read_text()}); give each read its own directory")
    record.write_text(json.dumps(meta))
    labels = torch.randint(0, 1000, (args.n,), generator=torch.Generator().manual_seed(args.seed)).tolist()
    base = args.shard * args.n // args.num_shards
    shard_labels = labels[base:(args.shard + 1) * args.n // args.num_shards]
    print(f"{args.schedule}, {calls} calls: shard {args.shard}/{args.num_shards}, {len(shard_labels)} samples "
          f"from index {base}", flush=True)
    vae = load_vae(dev)
    done = 0
    while done < len(shard_labels):
        m = len(shard_labels[done:done + args.bs])
        if all((sdir / f"{base + done + j:06d}.png").exists() for j in range(m)):
            done += m
            continue
        y = torch.tensor(shard_labels[done:done + m], device=dev)
        gens = [torch.Generator().manual_seed(args.seed * 1_000_003 + base + done + j) for j in range(m)]
        z = torch.stack([torch.randn(4, 32, 32, generator=g) for g in gens]).to(dev)

        def noise(shape, gens=gens):
            return torch.stack([torch.randn(shape[1:], generator=g) for g in gens]).to(dev)

        if args.schedule == "sequential":
            out = walk_sequential(model, z, y, args.groups, args.group_levels, levels, noise)
        else:
            out = walk(model, z, y, knots, noise)
        lat = out.float() / SCALE
        for i0 in range(0, m, args.decode_bs):
            with torch.no_grad():
                img = vae.decode(lat[i0:i0 + args.decode_bs]).sample
            img = ((img.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for j in range(img.shape[0]):
                Image.fromarray(img[j]).save(sdir / f"{base + done + i0 + j:06d}.png")
        done += m
        print(f"  sampled {done}/{len(shard_labels)}", flush=True)
    print(f"SHARD_DONE {args.shard}", flush=True)


def score(args):
    sdir = Path(args.sample_dir)
    metas = [json.loads(p.read_text()) for p in sorted(sdir.glob("shard_*.json"))]
    if not metas or len(metas) != metas[0]["num_shards"] or any(m != metas[0] for m in metas):
        raise SystemExit(f"{sdir}: the shards' records are missing or disagree")
    meta = metas[0]
    if meta["n"] != args.n:
        raise SystemExit(f"{sdir} holds a read of {meta['n']} samples, not {args.n}")
    st = adm_score(sdir, args.n, args.adm_dir)
    with Rows(args.name, args.out) as rows:
        rows.add(measure=meta["measure"], schedule=meta["schedule"], calls=meta["calls"], step=meta["step"], samples=args.n,
                 seed=meta["seed"], fid=st["FID"], sfid=st["sFID"], inception_score=st["Inception Score"],
                 precision=st["Precision"], recall=st["Recall"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["sample", "score"])
    ap.add_argument("--sample-dir", required=True, help="where the samples go, one directory per cell")
    ap.add_argument("--n", type=int, default=50000, help="samples in the cell")
    ap.add_argument("--ckpt", default="", help="sample: the model's checkpoint (train_imagenet)")
    ap.add_argument("--schedule", default="synchronised", choices=["synchronised", "sequential"])
    ap.add_argument("--calls", type=int, default=0,
                    help="sample, synchronised: calls on evenly spaced levels (0: every level, 249)")
    ap.add_argument("--groups", type=int, default=16, help="sample, sequential: groups of patches")
    ap.add_argument("--group-levels", type=int, default=16, help="sample, sequential: levels a group")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--bs", type=int, default=125, help="sample: samples a batch")
    ap.add_argument("--decode-bs", type=int, default=32, help="sample: latents a VAE decode")
    ap.add_argument("--adm-dir", default="", help="score: setup/prepare_adm_evaluator.sh's directory")
    ap.add_argument("--out", default="", help="score: where the row goes")
    ap.add_argument("--name", default="", help="score: the cell's name, OUT/NAME.csv")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if args.command == "sample":
        if not args.ckpt:
            ap.error("sample needs --ckpt")
        sample(args)
    else:
        if not (args.adm_dir and args.out and args.name):
            ap.error("score needs --adm-dir, --out and --name")
        score(args)


if __name__ == "__main__":
    main()
