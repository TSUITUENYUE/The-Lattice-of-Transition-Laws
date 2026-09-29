"""Dependence among masked text8 coordinates, measured on a trained model before any decode.

    python -m lattice.experiments.text8_dependence kernel --ckpt CKPT --data DATA/text8 --out DIR \
        [--rates 0.3,0.5,0.7] [--pairs 512] [--channel masked|graded --stage class|character] \
        [--distances 1,2,4,8,16,32] [--between S,...]
    python -m lattice.experiments.text8_dependence triplet --ckpt CKPT --data DATA/text8 --out DIR \
        [--rates 0.3,0.5,0.7] [--triples 512]
    python -m lattice.experiments.text8_dependence sets --ckpt CKPT --data DATA/text8 --out DIR [--n 256]
    (all modes: [--name NAME] [--seed 0] [--bs 64])

Every mode reads random windows of 256 test characters, masks each position at a fixed rate (with
at least one masked position per window), and compares the product law of a set of masked
coordinates (one call) with the exact chain over the same set (one call per coordinate). The
difference is the set's total correlation per coordinate. kernel and triplet draw windows, masks
and positions from one CPU generator seeded with --seed, cell after cell; sets seeds a generator
per cell with seed + 7000 + 100 x rate + k, so both rules of a cell read the same windows.

Modes
  kernel   pairs of masked coordinates at distance d = 1, 2, 4, 8, 16, 32, split by whether the
           span between them holds a space (across a boundary) or not (inside one word); --pairs
           per cell (a cell with fewer than 64 is not read). `gap` is the per-coordinate gap, half
           the entropy decrease at one coordinate when the other is revealed: the kernel of
           Figure 4 (a) and Tables 10 and 12 is twice the gap. --channel graded reads the
           graded model one stage at a time: class, both coordinates at the mask and each charged
           its class (its character probabilities summed over the class); character, both at their
           classes and each charged its character (Table 22). --distances replaces the
           distances read. --between reads each pair once for every s listed, with exactly s of the
           d - 1 coordinates strictly between the pair revealed, placed uniformly, and the others
           masked (the rest of the window keeps the mask drawn at the rate; s > d - 1 is skipped);
           such rows also record `between` and `crossing_fraction`, the fraction of spans of d + 1
           characters holding a space among every seventh start in the split's first 200,000
           characters (Table 16, and the estimate with the state between of text8_estimate).
  triplet  consecutive triples a < b < c of masked coordinates with gaps d1 and d2 (1, 2, 4, 8,
           16), charged at c three times on one state: nothing revealed, b revealed, a and b
           revealed. `pair` is the first difference, `triplet` the second, I(x_c; x_a | x_b), both
           in bits at one coordinate; `cross_bc` is the fraction of spans b..c holding a space.
           The inputs of the second-order estimate (Table 15) and of the kernel's corpus
           weighting (text8_estimate).
  sets     sets of k = 8, 16, 32 masked coordinates chosen by the random or the spread rule, at
           mask rates 0.3, 0.5 and 0.7 (at least k + 2 masked positions per window), n windows per
           cell (Table 11).

Product, chain, gap, pair and triplet are written rounded to five decimals, the fractions to four.
Writes OUT/NAME_s<seed>.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from ..data.text8 import LN2, MASK, V, Graded, Text8, mask_batch, windows
from ..evaluation.text8_bpc import chain_bpc, forward_bf16, logits_at, product_bpc, select_sets, top_up
from ..models.text import load
from ..utils.results import Rows

RATES = (0.3, 0.5, 0.7)
KERNEL_DIST = (1, 2, 4, 8, 16, 32)
TRIPLET_DIST = (1, 2, 4, 8, 16)
GROUPS = ("within", "crossing")
WIDTHS = (8, 16, 32)
SET_RULES = ("random", "spread")


def space_id(split):
    """The most frequent character of the split's first 200,000: the space."""
    return int(np.bincount(np.asarray(split[:200000])).argmax())


# ---- kernel ----------------------------------------------------------------------------------
def collect_pairs(split, u0, d, target, seq, space, gen, max_draws=400, between=None):
    """Pairs (window, masked window, left position) of masked coordinates at distance d, by group;
    with ``between`` = s, exactly s of the d - 1 coordinates strictly between them revealed, placed
    uniformly, the others masked."""
    buf = {g: [] for g in GROUPS}
    draws = 0
    while draws < max_draws and any(len(buf[g]) < target for g in GROUPS):
        draws += 1
        x = windows(split, 128, seq, gen)
        u = torch.full((128,), u0).clamp(min=4.0 / seq + 0.02)
        xin, m = mask_batch(x, u, gen)
        pair_ok = m[:, :-d] & m[:, d:]
        for r in range(x.shape[0]):
            cand = torch.nonzero(pair_ok[r], as_tuple=False).flatten()
            if cand.numel() == 0:
                continue
            i = int(cand[int(torch.randint(0, cand.numel(), (1,), generator=gen))])
            g = "crossing" if bool((x[r, i:i + d + 1] == space).any()) else "within"
            if len(buf[g]) < target:
                xr = xin[r]
                if between is not None:
                    xr = xr.clone()
                    inner = torch.arange(i + 1, i + d)
                    shown = inner[torch.randperm(d - 1, generator=gen)[:between]]
                    xr[inner] = MASK
                    xr[shown] = x[r, shown]
                buf[g].append((x[r], xr, i))
    return buf


def charge_pairs(net, items, d, bs, device):
    """Mean product and chain bits per coordinate over the pairs."""
    tot_p = tot_c = 0.0
    n = 0
    for k in range(0, len(items), bs):
        chunk = items[k:k + bs]
        x = torch.stack([c[0] for c in chunk]).to(device)
        xin = torch.stack([c[1] for c in chunk]).to(device)
        S = torch.tensor([[c[2], c[2] + d] for c in chunk]).to(device)
        tok = x.gather(1, S)
        p = product_bpc(net, xin, S, tok)
        c = chain_bpc(net, xin, S, tok)
        tot_p += p * len(chunk)
        tot_c += c * len(chunk)
        n += len(chunk)
    return tot_p / n, tot_c / n, n


@torch.no_grad()
def charge_graded(net, channel, items, d, bs, stage, device):
    """Mean product and chain bits per coordinate over the pairs on the graded channel. At the
    class stage both coordinates sit at the mask and each is charged its class; at the character
    stage both sit at their classes and each is charged its character. The chain shows the first
    coordinate's value at that stage before charging the second. Context positions are masked or
    show their characters."""
    classes = channel.classes.to(device)
    up = 1 if stage == "class" else channel.levels - 1
    tot_p = tot_c = 0.0
    n = 0
    for k in range(0, len(items), bs):
        chunk = items[k:k + bs]
        x = torch.stack([c[0] for c in chunk]).to(device)
        masked = torch.stack([c[1] for c in chunk]).to(device) == MASK
        rows = torch.arange(len(chunk)).to(device)
        i = torch.tensor([c[2] for c in chunk]).to(device)
        j = i + d
        level = torch.where(masked, torch.zeros_like(x), torch.full_like(x, channel.levels - 1))
        if stage == "character":
            level[rows, i] = 1
            level[rows, j] = 1

        def cost(lv, pos):
            lg = forward_bf16(net, channel.render(x, lv))
            lp = lg[rows, pos, :V].float().log_softmax(-1)
            tok = x[rows, pos]
            if stage == "class":
                same = classes.view(1, -1) == classes[tok].view(-1, 1)
                return -torch.logsumexp(lp.masked_fill(~same, float("-inf")), -1)
            return -lp.gather(1, tok.view(-1, 1)).squeeze(1)
        ci, cj = cost(level, i), cost(level, j)
        shown = level.clone()
        shown[rows, i] = up
        cj2 = cost(shown, j)
        tot_p += float((ci + cj).sum()) / 2
        tot_c += float((ci + cj2).sum()) / 2
        n += len(chunk)
    return tot_p / n / LN2, tot_c / n / LN2, n


def crossing_fraction(split, space, d):
    """The fraction of spans of d + 1 characters holding a space, over every seventh start in the
    split's first 200,000 characters."""
    arr = np.asarray(split[:200000])
    return float(np.mean([bool((arr[i:i + d + 1] == space).any()) for i in range(0, len(arr) - d - 1, 7)]))


def mode_kernel(args, net, split, rows):
    space = space_id(split)
    channel = Graded(levels=3) if args.channel == "graded" else None
    gen = torch.Generator().manual_seed(args.seed)
    dists = [int(v) for v in args.distances.split(",")]
    betweens = [int(v) for v in args.between.split(",")] if args.between else [None]
    frac = {d: crossing_fraction(split, space, d) for d in dists} if args.between else {}
    for u0 in [float(r) for r in args.rates.split(",")]:
        for d in dists:
            for s in betweens:
                if s is not None and s > d - 1:
                    continue
                buf = collect_pairs(split, u0, d, args.pairs, args.seq, space, gen, between=s)
                for g in GROUPS:
                    if len(buf[g]) < 64:
                        print(f"rate={u0} d={d} between={s} group={g}: {len(buf[g])} pairs, too few; not read",
                              flush=True)
                        continue
                    p, c, n = (charge_pairs(net, buf[g], d, args.bs, args.device) if channel is None else
                               charge_graded(net, channel, buf[g], d, args.bs, args.stage, args.device))
                    extra = {} if s is None else dict(between=s, crossing_fraction=round(frac[d], 5))
                    rows.add(rate=u0, d=d, **extra, group=g, pairs=n, product=round(p, 5), chain=round(c, 5),
                             gap=round(p - c, 5))


# ---- triplet ---------------------------------------------------------------------------------
def collect_triples(split, u0, d1, d2, target, seq, gen, max_draws=400):
    """Triples (window, masked window, a) of masked coordinates a < a + d1 < a + d1 + d2."""
    out, draws, span = [], 0, d1 + d2
    while draws < max_draws and len(out) < target:
        draws += 1
        x = windows(split, 128, seq, gen)
        u = torch.full((128,), u0).clamp(min=4.0 / seq + 0.02)
        xin, m = mask_batch(x, u, gen)
        ok = m[:, :seq - span] & m[:, d1:seq - d2] & m[:, span:]
        for r in range(x.shape[0]):
            cand = torch.nonzero(ok[r], as_tuple=False).flatten()
            if cand.numel() == 0:
                continue
            i = int(cand[int(torch.randint(0, cand.numel(), (1,), generator=gen))])
            out.append((x[r], xin[r], i))
            if len(out) >= target:
                break
    return out


@torch.no_grad()
def ce_at(net, xin, pos, tok):
    """Cross-entropy in bits at one coordinate per row."""
    lg = logits_at(net, xin, pos.view(-1, 1)).squeeze(1)
    return F.cross_entropy(lg.float(), tok, reduction="none") / LN2


def charge_triples(net, items, d1, d2, bs, space, device):
    """Per-triple pair and triplet terms at c, and the fractions of spans holding a space."""
    pair, trip, cs, cb = [], [], [], []
    for k in range(0, len(items), bs):
        chunk = items[k:k + bs]
        x = torch.stack([t[0] for t in chunk]).to(device)
        x0 = torch.stack([t[1] for t in chunk]).to(device)
        a = torch.tensor([t[2] for t in chunk], device=device)
        b, c = a + d1, a + d1 + d2
        rows = torch.arange(len(chunk), device=device)
        tok = x[rows, c]
        x1 = x0.clone()
        x1[rows, b] = x[rows, b]
        x2 = x1.clone()
        x2[rows, a] = x[rows, a]
        e0, e1, e2 = ce_at(net, x0, c, tok), ce_at(net, x1, c, tok), ce_at(net, x2, c, tok)
        pair.append((e0 - e1).cpu())
        trip.append((e1 - e2).cpu())
        ar = torch.arange(d1 + d2 + 1, device=device).view(1, -1)
        cs.append((x.gather(1, a.view(-1, 1) + ar) == space).any(1).float().cpu())
        br = torch.arange(d2 + 1, device=device).view(1, -1)
        cb.append((x.gather(1, b.view(-1, 1) + br) == space).any(1).float().cpu())
    return torch.cat(pair), torch.cat(trip), float(torch.cat(cs).mean()), float(torch.cat(cb).mean())


def sem(v):
    return float(v.std() / len(v) ** 0.5)


def mode_triplet(args, net, split, rows):
    space = space_id(split)
    gen = torch.Generator().manual_seed(args.seed)
    for u0 in [float(r) for r in args.rates.split(",")]:
        for d1 in TRIPLET_DIST:
            for d2 in TRIPLET_DIST:
                items = collect_triples(split, u0, d1, d2, args.triples, args.seq, gen)
                if len(items) < 64:
                    print(f"rate={u0} d1={d1} d2={d2}: {len(items)} triples, too few; not read", flush=True)
                    continue
                p, t, fs, fb = charge_triples(net, items, d1, d2, args.bs, space, args.device)
                rows.add(rate=u0, d1=d1, d2=d2, triples=len(items), pair=round(float(p.mean()), 5),
                         pair_sem=round(sem(p), 5), triplet=round(float(t.mean()), 5),
                         triplet_sem=round(sem(t), 5), cross_span=round(fs, 4), cross_bc=round(fb, 4))


# ---- sets ------------------------------------------------------------------------------------
def mode_sets(args, net, split, rows):
    for u0 in RATES:
        for k in WIDTHS:
            for rule in SET_RULES:
                # one generator per cell, so the two rules see the same windows and masks
                g = torch.Generator().manual_seed(args.seed + 7000 + int(u0 * 100) + k)
                x = windows(split, args.n, args.seq, g)
                u = torch.full((args.n,), u0).clamp(min=(k + 2) / args.seq + 0.02)
                xin, m = mask_batch(x, u, g)
                xin, m = top_up(xin, m, k, g)
                S = select_sets(m, k, rule=rule, generator=g)
                x, xin, S = x.to(args.device), xin.to(args.device), S.to(args.device)
                tok = x.gather(1, S)
                P = product_bpc(net, xin, S, tok)
                J = chain_bpc(net, xin, S, tok)
                rows.add(rate=u0, k=k, rule=rule, product=P, chain=J, gap=P - J,
                         meandist=float((S[:, 1:] - S[:, :-1]).float().mean()))


MODES = {"kernel": mode_kernel, "triplet": mode_triplet, "sets": mode_sets}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=list(MODES))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="the directory of the text8 split (test.bin)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="", help="the stem of the output file (default: the mode)")
    ap.add_argument("--rates", default=",".join(map(str, RATES)), help="kernel, triplet: mask rates")
    ap.add_argument("--pairs", type=int, default=512, help="kernel: pairs per cell")
    ap.add_argument("--triples", type=int, default=512, help="triplet: triples per cell")
    ap.add_argument("--n", type=int, default=256, help="sets: windows per cell")
    ap.add_argument("--channel", default="masked", choices=["masked", "graded"],
                    help="kernel: graded reads the graded model one stage at a time")
    ap.add_argument("--stage", default="class", choices=["class", "character"], help="kernel, graded: the stage")
    ap.add_argument("--distances", default=",".join(map(str, KERNEL_DIST)), help="kernel: the distances read")
    ap.add_argument("--between", default="", help="kernel: revealed coordinates between the pair, comma "
                    "separated (empty: the span as the mask drew it)")
    ap.add_argument("--seq", type=int, default=256, help="window length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=64, help="rows per call")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    net, meta = load(args.ckpt, kind="masked", device=args.device)
    split = Text8(args.data).test
    name = args.name or args.mode
    with Rows(name, args.out, meta=meta, seed=args.seed) as rows:
        MODES[args.mode](args, net, split, rows)


if __name__ == "__main__":
    main()
