"""One text8 model read at both corners of the lattice (the token block of Figure 2).

    python -m lattice.experiments.text8_corners --ckpt CKPT --kind causal|masked|two-stream \
        --data DATA/text8 --out DIR [--name NAME] [--prefix 192] [--limit 1000000] [--n 512] \
        [--draws 32]

The sequential corner is the exact chain in identity order at long context: windows of 256 test
characters overlapping so that only the last 64 of each are scored, the 192 before them revealed,
over the first --limit characters of the test split (evaluation/text8_bpc.py, sliding_chain_bpc).
A causal model reads it through its prefixes, a masked or lattice model (--kind masked) through
mask tokens after the revealed prefix, the two-stream model under the identity order.

The synchronised corner is the masked ELBO on n windows of 256 test characters evenly spaced over
the test split: --draws draws of a mask rate per window (a uniform clamped to [0.01, 1]) and a mask
at that rate (no position forced), every masked position's cross-entropy weighted by the
reciprocal of its window's rate, per character. Every model reads the same windows, rates and
masks. The two-stream model presents a masked state through its query stream, every revealed
coordinate at rank zero and every masked one at rank one, one window per call. A causal model
cannot present a masked state, so it has no ELBO row.

Writes OUT/NAME_s0.csv.
"""
from __future__ import annotations

import argparse

import torch

from ..data.text8 import Text8, fixed_windows
from ..evaluation.text8_bpc import mdm_elbo_bpc, sliding_chain_bpc, two_stream_elbo_bpc
from ..models.text import load
from ..utils.results import Rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kind", required=True, choices=["causal", "masked", "two-stream"],
                    help="how the model reads: causal, masked (the masked and lattice models) or two-stream")
    ap.add_argument("--data", required=True, help="the directory of the text8 split (test.bin)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="text8_corners", help="the stem of the output file")
    ap.add_argument("--seq", type=int, default=256, help="window length")
    ap.add_argument("--prefix", type=int, default=192, help="revealed characters in front of the scored tail")
    ap.add_argument("--limit", type=int, default=1_000_000, help="test characters the chain reads")
    ap.add_argument("--n", type=int, default=512, help="windows of the masked ELBO")
    ap.add_argument("--draws", type=int, default=32, help="draws of the masked ELBO")
    ap.add_argument("--bs", type=int, default=64, help="windows per call")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    net, meta = load(args.ckpt, kind=args.kind, device=args.device)
    split = Text8(args.data).test
    model = net
    if args.kind == "two-stream":
        rank = torch.arange(args.seq)
        model = lambda z: net(z, rank)          # the identity order: next-character prediction
    with Rows(args.name, args.out, meta=meta, seed=0) as rows:
        chain = sliding_chain_bpc(model, split, seq=args.seq, prefix=args.prefix, bs=args.bs,
                                  limit=args.limit, device=args.device)
        rows.add(corner="sequential", reading="exact chain", prefix=args.prefix, bpc=chain)
        if args.kind == "causal":
            return
        windows = [w for w in fixed_windows(split, args.n, args.seq).numpy()]
        elbo = two_stream_elbo_bpc if args.kind == "two-stream" else mdm_elbo_bpc
        rows.add(corner="synchronised", reading="masked ELBO",
                 bpc=elbo(net, windows, bs=args.bs, draws=args.draws, device=args.device))


if __name__ == "__main__":
    main()
