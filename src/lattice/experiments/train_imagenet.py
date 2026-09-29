"""Train one SiT on ImageNet-256 latents, on one training distribution over the lattice.

    python -m lattice.experiments.train_imagenet --name NAME --ckpt-dir DIR \
        --latents DATA/imagenet/latents.pt --flip-bank DATA/imagenet/latents_flip.pt \
        --measure synchronised|mixed [--model SiT-B/2] [--steps 400000] [options]

The two models of Table 8 are this command with --measure changed and nothing else:
  --measure mixed          a level per patch: a quarter of each micro-batch at one common level, a
                           quarter on the patch-sequential path over the 256 patches in raster
                           order, half at independent levels ("lattice, mixed levels")
  --measure synchronised   one level per image, which is SiT's own training ("synchronised only")
(training/measures.py, the distributions the CIFAR-10 models train on). The rest is SiT's pipeline:
SiT-B/2 on the 32 x 32 x 4 latent (2 x 2 patches, 256 tokens), the linear interpolant
x_t = t x_1 + (1 - t) x_0 between noise x_0 and data x_1, the network regressing the noise x_0 at
one of the levels t = k / 250 with uniform weight, AdamW (learning rate 1e-4, no weight decay),
EMA 0.9999 per step, batch 256 in two micro-batches of 128, class dropout 0.1, bf16 autocast on a
GPU. The synchronised model draws k uniformly from 1 .. 249; the per-patch measures draw k from
0 .. 250 per patch and clamp it to 1 .. 249, since t = 0 and t = 1 carry no gradient. A draw
takes an image's latent from the plain or the mirrored bank with probability 1/2, and its latent
from the VAE posterior (mean and log-variance) times 0.18215.

A per-patch model first checks that a level constant across the tokens reproduces SiT's own
forward and that a varying one moves it (models/sit_levels.py). Writes DIR/ckpt_NAME.pt (EMA
weights and the configuration) and resumes from DIR/res_NAME.pt, saved every --save-every steps,
with every random-number state restored, so an interrupted run matches an uninterrupted one.
"""
from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path

import torch

from ..data.imagenet_latents import SCALE, read_bank
from ..models import sit
from ..models import sit_levels as SL
from ..training.measures import MEASURES, level_drawer


@torch.no_grad()
def update_ema(ema, model, decay=0.9999):
    ep = dict(ema.named_parameters())
    for name, p in model.named_parameters():
        ep[name].mul_(decay).add_(p.data, alpha=1 - decay)


def _save(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="the model's name: DIR/ckpt_NAME.pt")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--latents", required=True, help="the latent bank: a packed file or a directory of shards")
    ap.add_argument("--flip-bank", default="", help="the index-aligned bank of the mirrored images")
    ap.add_argument("--measure", default="synchronised", choices=MEASURES, help="the training distribution")
    ap.add_argument("--model", default="SiT-B/2", choices=sorted(sit.SiT_models))
    ap.add_argument("--steps", type=int, default=400000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--micro-bs", type=int, default=128, help="gradient accumulation in pieces of this size")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--levels", type=int, default=250, help="the grid t = k / levels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=5000, help="steps between rolling states")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args(argv)
    if args.bs % args.micro_bs:
        ap.error("--bs must be a multiple of --micro-bs")
    naccum = args.bs // args.micro_bs
    dev = args.device
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    outdir = Path(args.ckpt_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    MU, LV, CL = read_bank(args.latents)
    N = len(MU)
    MU2 = LV2 = None
    if args.flip_bank:
        MU2, LV2, _ = read_bank(args.flip_bank, labels=False)
        if len(MU2) != N:
            raise SystemExit(f"the flip bank holds {len(MU2)} latents, the bank {N}")
    print(f"{args.name}: {N} latents, {int(CL.max()) + 1} classes, flip bank "
          f"{'yes' if MU2 is not None else 'no'}, {args.model}, measure {args.measure}", flush=True)

    model = sit.SiT_models[args.model](input_size=32, num_classes=1000).to(dev)
    n_patches = model.x_embedder.num_patches
    side = int(n_patches ** 0.5)
    pside = model.x_embedder.patch_size[0]
    perpatch = args.measure != "synchronised"
    if perpatch:
        gap, moved, synthetic = SL.check_constant_path(model, dev, tokens=n_patches)
        print(f"per-patch path: {n_patches} patches, agreement {gap:.2e} and movement {moved:.3f} "
              f"relative to the output scale" + (" (on a copy with the modulation switched on)"
                                                 if synthetic else ""), flush=True)
        if not (gap < 1e-4 and moved > 1e-3):
            raise SystemExit(f"the per-patch path fails its check: agreement {gap}, movement {moved}")
        draw = level_drawer(args.measure, args.levels, n_patches, gen, dev)
    ema = copy.deepcopy(model).to(dev)
    for p in ema.parameters():
        p.requires_grad_(False)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    res = outdir / f"res_{args.name}.pt"
    start = 0
    if args.resume and res.exists():
        # loaded to the CPU: load_state_dict moves weights and optimiser state to the model's
        # device, and every random-number state is restored from a CPU tensor
        rs = torch.load(res, map_location="cpu", weights_only=False)
        model.load_state_dict(rs["model"])
        ema.load_state_dict(rs["ema"])
        opt.load_state_dict(rs["opt"])
        start = rs["step"]
        gen.set_state(rs["rng"]["generator"].cpu())
        torch.set_rng_state(rs["rng"]["torch"].cpu())
        if rs["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in rs["rng"]["cuda"]])
        print(f"{args.name}: resumed at step {start}", flush=True)
    meta = {"arch": args.model, "measure": args.measure, "levels": args.levels, "seed": args.seed,
            "bs": args.bs, "lr": args.lr, "flip_bank": bool(args.flip_bank)}

    model.train()
    t0 = time.time()
    for it in range(start, args.steps):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(naccum):
            idx = torch.randint(0, N, (args.micro_bs,), generator=gen)
            y = CL[idx].to(dev)
            if MU2 is not None:
                flc = (torch.rand(args.micro_bs, generator=gen) < 0.5).view(-1, 1, 1, 1)
                mu = torch.where(flc, MU2[idx], MU[idx]).to(dev, torch.float32)
                lv = torch.where(flc, LV2[idx], LV[idx]).to(dev, torch.float32)
            else:
                mu = MU[idx].to(dev, torch.float32)
                lv = LV[idx].to(dev, torch.float32)
            x1 = (mu + (0.5 * lv).exp() * torch.randn(mu.shape, generator=gen).to(dev)) * SCALE
            x0 = torch.randn(x1.shape, generator=gen).to(dev)
            if perpatch:
                t = draw(args.micro_bs).clamp(1, args.levels - 1).float() / args.levels
                te = SL.levels_to_pixels(t, side, pside)
            else:
                t = torch.randint(1, args.levels, (args.micro_bs,), generator=gen).to(dev).float() / args.levels
                te = t.view(-1, 1, 1, 1)
            xt = te * x1 + (1 - te) * x0
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                d0 = SL.forward_perpatch(model, xt, t, y) if perpatch else model(xt, t, y)
                loss = ((d0.float() - x0) ** 2).mean() / naccum
            loss.backward()
            tot += float(loss.detach())
        opt.step()
        update_ema(ema, model)
        if it % args.log_every == 0:
            print(f"LOSS it={it} loss={tot:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if (it + 1) % args.save_every == 0:
            _save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                   "step": it + 1, "rng": {"generator": gen.get_state(), "torch": torch.get_rng_state(),
                                           "cuda": torch.cuda.get_rng_state_all()
                                           if torch.cuda.is_available() else None}, **meta}, res)
    _save({"ema": ema.state_dict(), "step": args.steps, "tag": args.name, **meta},
          outdir / f"ckpt_{args.name}.pt")
    print(f"TRAIN_DONE {args.name}", flush=True)


if __name__ == "__main__":
    main()
