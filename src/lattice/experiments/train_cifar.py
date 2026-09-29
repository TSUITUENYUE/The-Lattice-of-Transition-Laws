"""Train one CIFAR-10 model with a noise level per patch, on one training distribution over the
lattice.

    python -m lattice.experiments.train_cifar --name NAME --ckpt-dir DIR --data-root DATA/cifar10 \
        --measure iid|mixed|synchronised|sequential [--steps 60000] [--continue-from RES] [options]

The paper's four CIFAR-10 models are this command with --measure changed and nothing else, so they
share the network, data, levels, seed and budget (60,000 steps at batch 512):
  --measure iid            every patch at an independent uniform level ("iid levels")
  --measure mixed          a quarter of each batch at the synchronised corner, a quarter on the
                           patch-sequential path, half at independent levels ("mixed levels")
  --measure synchronised   one level for the whole image, a diffusion model ("synchronised only")
  --measure sequential     states of the patch-sequential path only ("sequential only")
(training/measures.py). The network is EDM's class-conditional SongUNet at width 128 with the
level map as an extra input channel and EDM's preconditioning applied per pixel, on 16 patches of
8 by 8 (models/pixel.py); the levels are geometric in sigma from 80 to 0.002 over --transitions
(core/levels.py); the recipe is EDM's (training/edm.py).

Writes DIR/ckpt_NAME.pt (EMA weights and the configuration) and resumes from DIR/res_NAME.pt with
every random-number state restored. --continue-from starts from another training's rolling state
under this name: the mixed model carried from 60,000 to 120,000 steps is --continue-from
DIR/res_mixed.pt --steps 120000, which draws the data an uninterrupted 120,000-step run would draw.
"""
from __future__ import annotations

import argparse
import math

import torch

from ..core.levels import edm_levels
from ..data.cifar10 import Cifar
from ..models import pixel as PX
from ..training import edm as TR
from ..training.measures import MEASURES, level_drawer


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="the model's name: DIR/ckpt_NAME.pt")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--data-root", required=True, help="the directory setup/prepare_cifar10.sh filled")
    ap.add_argument("--measure", default="iid", choices=MEASURES, help="the training distribution")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--micro-bs", type=int, default=128, help="gradient accumulation in pieces of this size")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--channels", type=int, default=128, help="the U-Net's base width")
    ap.add_argument("--patch", type=int, default=8, help="side of a patch that shares one level")
    ap.add_argument("--transitions", type=int, default=128)
    ap.add_argument("--sigma-max", type=float, default=80.0)
    ap.add_argument("--sigma-min", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=5000, help="steps between rolling states")
    ap.add_argument("--continue-from", default="",
                    help="another training's rolling state (res_*.pt) to start from")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args(argv)
    dev = args.device
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    cifar = Cifar(args.data_root, device=dev)
    levels = edm_levels(args.transitions, args.sigma_max, args.sigma_min)
    knots = torch.tensor([math.sqrt(float(k)) for k in levels.knots], device=dev)
    patches, n_patches = PX.patch_index(32, args.patch)
    patches = patches.to(dev)

    model = PX.LevelMapEDM(channels=args.channels).to(dev)
    ema = PX.LevelMapEDM(channels=args.channels).to(dev)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    npar = sum(p.numel() for p in model.parameters())
    print(f"{args.name}: {npar / 1e6:.1f}M params, {n_patches} patches, measure={args.measure}", flush=True)

    draw = level_drawer(args.measure, args.transitions, n_patches, gen, dev)

    def loss_fn(net, x0, onehot):
        return TR.edm_loss(net, x0, onehot, PX.level_map(draw(x0.shape[0]), patches, knots))

    TR.run(model, ema, cifar, args.steps, args.ckpt_dir, args.name, loss_fn=loss_fn, generator=gen,
           lr=args.lr, bs=args.bs, micro_bs=args.micro_bs, resume=args.resume,
           save_every=args.save_every, start_from=args.continue_from or None,
           extra_state={"channels": args.channels, "patch": args.patch,
                        "transitions": args.transitions, "measure": args.measure,
                        "params": npar, "seed": args.seed})


if __name__ == "__main__":
    main()
