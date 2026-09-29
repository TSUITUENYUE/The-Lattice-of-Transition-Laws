"""Advances from lattice states read under each architecture (Table 9).

    python -m lattice.experiments.text8_readable --causal CKPT --bidirectional CKPT \
        --two-stream CKPT --data DATA/text8 --out DIR [--seed 0] [--n 512] [--rate 0.5]

An advance is a conditioning state and an advanced set. Every architecture is read under the law
it expresses in one call: the causal model (prefix states, width one only) and the two-stream
model (any state, a sequential chain over the advanced set, since a query reads the coordinates
of lower rank) and the bidirectional model, the dedicated masked model (any state, the product
law over the set). A row an architecture cannot express is written with an empty reading.

The four advances: prefix, the first 128 of 256 characters revealed and the next 1 or 8 advanced;
interior, every position revealed with probability 1 - rate and the advanced set chosen among the
masked positions by the spread rule, width 1 or 8. With width one the spread rule takes the
leftmost masked position. The windows (random test windows) and the conditioning pattern are drawn
from a CPU generator seeded with seed + width (+ 1000 for interior); one pattern is shared by the
batch, so one reveal order describes the state, and every architecture reads the same state. The
two-stream model reads the revealed coordinates in identity order for a prefix state and in a
random order for an interior state (the two orders it was trained on), then the advanced set in
position order, then the rest in a random order.

Writes OUT/text8_readable_s<seed>.csv.
"""
from __future__ import annotations

import argparse

import torch

from ..core import selection as SEL
from ..data.text8 import MASK, Text8, windows
from ..evaluation.text8_bpc import product_bpc
from ..models.text import load
from ..utils.results import Rows

POINTS = (("prefix, width one", "prefix", 1), ("prefix, width eight", "prefix", 8),
          ("interior, width one", "interior", 1), ("interior, width eight", "interior", 8))


def expresses(arch, kind, width):
    return arch != "causal" or (kind == "prefix" and width == 1)


def rank_of(revealed, advanced, seq, generator=None, internal="random"):
    """The reveal order of a state: the revealed coordinates first (in identity or random order),
    then the advanced set in position order, then the rest in random order."""
    rank = torch.empty(seq, dtype=torch.long)
    n_cond = int(revealed.sum())
    n_adv = int(advanced.sum())
    if internal == "identity":
        rank[revealed] = torch.arange(n_cond)
    else:
        rank[revealed] = torch.randperm(n_cond, generator=generator)
    rank[advanced] = n_cond + torch.arange(n_adv)
    rest = ~revealed & ~advanced
    rank[rest] = n_cond + n_adv + torch.randperm(int(rest.sum()), generator=generator)
    return rank


def point_batch(kind, k, split, n, seq, rate, g, device):
    """The masked state, the clean windows, the advanced set, its targets and the reveal order."""
    x = windows(split, n, seq, g)
    revealed = torch.zeros(seq, dtype=torch.bool)
    if kind == "prefix":
        cut = seq // 2
        revealed[:cut] = True
        S = torch.arange(cut, cut + k).expand(n, -1).contiguous()
    else:
        keep = torch.rand(seq, generator=g) >= rate
        revealed |= keep
        masked = ~revealed
        if int(masked.sum()) < k:
            revealed[torch.randperm(seq, generator=g)[:k]] = False
            masked = ~revealed
        S = SEL.spread(masked, k).expand(n, -1).contiguous()
    advanced = torch.zeros(seq, dtype=torch.bool)
    advanced[S[0]] = True
    xin = x.clone()
    xin[:, ~revealed] = MASK
    x, xin, S = x.to(device), xin.to(device), S.to(device)
    rank = rank_of(revealed, advanced, seq, g, internal="identity" if kind == "prefix" else "random")
    return xin, x, S, x.gather(1, S), rank


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--causal", default="", help="the dedicated causal model")
    ap.add_argument("--bidirectional", default="", help="the dedicated masked model")
    ap.add_argument("--two-stream", default="", help="the two-stream model")
    ap.add_argument("--data", required=True, help="the directory of the text8 split (test.bin)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=256, help="window length")
    ap.add_argument("--n", type=int, default=512, help="windows per advance")
    ap.add_argument("--rate", type=float, default=0.5, help="the mask rate of an interior state")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    split = Text8(args.data).test
    models = []
    for arch, path, kind in (("causal", args.causal, "causal"), ("bidirectional", args.bidirectional, "masked"),
                             ("two-stream", args.two_stream, "two-stream")):
        if path:
            net, meta = load(path, kind=kind, device=args.device)
            models.append((arch, net, meta))
    if not models:
        raise SystemExit("name at least one architecture's model")
    with Rows("text8_readable", args.out, meta={a: m.get("step") for a, _, m in models}, seed=args.seed) as rows:
        for name, kind, k in POINTS:
            g = torch.Generator().manual_seed(args.seed + k + (0 if kind == "prefix" else 1000))
            xin, clean, S, tok, rank = point_batch(kind, k, split, args.n, args.seq, args.rate, g, args.device)
            for arch, net, _ in models:
                if not expresses(arch, kind, k):
                    rows.add(point=name, arch=arch, width=k, reading="none", bpc=float("nan"))
                    continue
                if arch == "two-stream":
                    # the state is the order: the sequence goes in clean and attention carries
                    # the conditioning
                    r = rank.to(args.device)
                    model, state = (lambda z, net=net, r=r: net(z, r)), clean
                else:
                    model, state = net, xin
                rows.add(point=name, arch=arch, width=k,
                         reading="chain" if arch == "two-stream" else "product",
                         bpc=product_bpc(model, state, S, tok))


if __name__ == "__main__":
    main()
