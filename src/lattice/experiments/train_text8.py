"""Train one text8 model.

    python -m lattice.experiments.train_text8 --model MODEL --data DATA_ROOT/text8 \\
        --ckpt-dir CKPT_ROOT/text8 --out OUTPUT_ROOT/train_text8 [options]

MODEL is one of
  causal      the dedicated causal model (next-character cross-entropy, rotary network)
  masked      the dedicated masked model (random-subset masking at a rate drawn per row)
  two-stream  the two-stream model (every conditional along a sampled reveal order)
  lattice     the lattice model (prefix states, random subsets and advanced sets in thirds)
  graded      the graded model of the channel pair (learned-position network)
  absorbing   the absorbing model of the channel pair (learned-position network)
training/text8.py states each objective and the checkpoint selection exactly.

The defaults are the paper's 85M configuration: width 768, 12 layers, 12 heads, windows of 256
characters, batch 512, dropout 0.2, seed 0; the step count, learning rate, micro-batch,
validation interval and validation windows default to each model's own (DEFAULTS below). The
26M lattice model is --model lattice --network learned --dim 512 --depth 8 --heads 8 --bs 255
--micro-bs 255 --steps 200000 --lr 3e-4 --dropout 0, read at its last step.

Writes to --ckpt-dir <name>_best.pt (the averaged weights at the validation minimum, with the
step they were banked at), <name>_final.pt (the averaged weights after the last step) and
<name>_state.pt (the rolling state a rerun resumes from); a model whose _final.pt exists is not
retrained. Every validation reading is appended to --out/train_<name>.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..models.text import MODELS, NETWORKS
from ..training.text8 import train_model

# per model: steps, learning rate, sequences per forward pass, validation interval and windows
DEFAULTS = {
    "causal": dict(steps=200000, lr=3e-4, micro_bs=64, val_every=5000, val_windows=512),
    "masked": dict(steps=200000, lr=3e-4, micro_bs=64, val_every=5000, val_windows=512),
    "two-stream": dict(steps=200000, lr=1e-4, micro_bs=64, val_every=5000, val_windows=256),
    "lattice": dict(steps=600000, lr=1.5e-4, micro_bs=256, val_every=10000, val_windows=512),
    "graded": dict(steps=200000, lr=3e-4, micro_bs=64, val_every=5000, val_windows=512),
    "absorbing": dict(steps=200000, lr=3e-4, micro_bs=64, val_every=5000, val_windows=512),
}


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--data", required=True,
                    help="directory with train.bin and valid.bin (setup/prepare_text8.sh)")
    ap.add_argument("--ckpt-dir", required=True, help="directory for the checkpoints")
    ap.add_argument("--out", required=True, help="directory for the validation log")
    ap.add_argument("--name", default="", help="file stem of the checkpoints [the model]")
    ap.add_argument("--network", choices=["rotary", "learned"], default=None,
                    help="lattice model only: rotary [default] or learned positions; the other "
                         "models train one network each (NETWORKS in models/text.py)")
    ap.add_argument("--dim", type=int, default=768, help="width")
    ap.add_argument("--depth", type=int, default=12, help="layers")
    ap.add_argument("--heads", type=int, default=12, help="attention heads")
    ap.add_argument("--seq", type=int, default=256, help="window length in characters")
    ap.add_argument("--bs", type=int, default=512, help="windows per optimiser step")
    ap.add_argument("--micro-bs", type=int, default=None,
                    help="windows per forward pass; bs / micro-bs are accumulated")
    ap.add_argument("--steps", type=int, default=None, help="optimiser steps")
    ap.add_argument("--lr", type=float, default=None, help="peak learning rate")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--val-every", type=int, default=None,
                    help="steps between validation readings (and banking)")
    ap.add_argument("--val-windows", type=int, default=None,
                    help="validation windows of the reading that banks the checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="start afresh even if a rolling state exists")
    args = ap.parse_args(argv)
    for k, v in DEFAULTS[args.model].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.network is not None and args.network not in NETWORKS[args.model]:
        ap.error(f"the {args.model} model trains the {NETWORKS[args.model][0]} network")
    if args.bs % args.micro_bs:
        ap.error("--bs must be a multiple of --micro-bs")
    return args


def main(argv=None):
    args = parse(argv)
    name = args.name or args.model
    if (Path(args.ckpt_dir) / f"{name}_final.pt").exists():
        print(f"{name}: finished model found in {args.ckpt_dir}; not retrained", flush=True)
        return
    train_model(args.model, args.data, args.ckpt_dir, name, network=args.network,
                dim=args.dim, depth=args.depth, heads=args.heads, seq=args.seq, bs=args.bs,
                micro_bs=args.micro_bs, steps=args.steps, lr=args.lr, dropout=args.dropout,
                val_every=args.val_every, val_windows=args.val_windows, seed=args.seed,
                resume=args.resume, device=args.device,
                log_path=Path(args.out) / f"train_{name}.csv")


if __name__ == "__main__":
    main()
