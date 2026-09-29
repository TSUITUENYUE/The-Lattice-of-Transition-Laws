"""Pairwise and second-order estimates of a schedule's dependence cost, from measured kernels.

    python -m lattice.experiments.text8_estimate fixed --structure GLOB --triplet GLOB --out DIR \
        [--name NAME] [--sets-device cuda]
    python -m lattice.experiments.text8_estimate decoded --structure GLOB --triplet GLOB \
        --sets FILE --out DIR [--name NAME]
    python -m lattice.experiments.text8_estimate channel --absorbing GLOB --class-kernel GLOB \
        --character-kernel GLOB --triplet GLOB --out DIR [--name NAME] [--sets-device cuda]
    python -m lattice.experiments.text8_estimate between --structure GLOB --out DIR [--name NAME] \
        [--sets-device cuda]

No model is read. The inputs are the kernel reads of text8_dependence: the pair reads (--structure,
the `kernel` mode) and the triplet reads (--triplet, the `triplet` mode), several seeds and mask
rates each (a glob may name files of several reads, space separated).

The kernel at mask rate r and distance d is twice the pair read's per-coordinate gap (the entropy
decrease at one coordinate when the other is revealed), pairs inside a word and pairs across a
word boundary weighted by the corpus fraction of spans with a space at that distance (read from
the triplet files), averaged over the seeds weighted by pairs. Written per rate read in both
inputs as NAME_kernel.csv (Table 12).

A schedule is walked step by step. A step's advanced set is sorted; every consecutive pair with no
revealed coordinate between them is charged the kernel at its distance, and every consecutive
triple with no revealed coordinate in its span is charged the triplet term I(x_c; x_a | x_b) at
its two gaps (the triplet read's `triplet`). Both terms are read at the step's own mask rate, the
fraction of the decoded region still masked before the step, interpolated linearly between the
rates read and held at the nearest rate outside them; in distance, linearly between the distances
read, and zero from distance 16 on for the pair term (beyond 16 for the triplet term). The
pairwise estimate is the pair charges summed over the steps and divided by the decoded length
(bits per character, Lemma 2's bound summed over the path); the second-order estimate adds the
triplet charges.

Modes
  fixed    the rules that read only positions, on the decoder's own sets: 64 characters after a
           192-character prefix at 2, 4, 8, 16 and 32 steps (text8_decode's protocol),
           contiguous, spread, separator and midpoint selected exactly as the decoder selects
           them, random
           averaged over 16 draws of its sets (the decoder draws one per window). Computed before
           any decode. spread orders positions equally near an anchor with the sort of the device
           it selects on, so --sets-device must name the device the decode ran on (the paper's
           decodes ran on a GPU: cuda).
  decoded  the sets a decode actually selected (text8_decode --save-sets), window by window,
           for the rules that read the model: confidence and min_distance. Computed after the
           decode.
  channel  the channel pair (Table 21): complete decodes of 256-character windows by the spread
           rule, each step at its own rate (the fraction of the window not yet at a character).
           The absorbing model's walk commits 256/T coordinates a step and is charged the absorbing
           kernel; the graded model's walk advances 512/T a step, revealing a set's classes and
           committing the same set at the next step, openings charged the class kernel and commits
           the character kernel given the classes; a pair is screened by a character committed
           between them. Pair terms only; the triplet reads (--triplet) give the mask rates and
           the corpus weighting of the two groups, as in the other modes.
  between  the first-order reference and the estimate with the state between (Table 17),
           from pair reads made with text8_dependence kernel --between (--structure; no triplet
           reads): the kernel at mask rate r, distance d and s revealed coordinates between the pair
           is twice the gap, weighted by pairs over the seeds, the two groups mixed at the read's
           own crossing_fraction where both were read. Every consecutive pair of a step's sorted set
           is charged the kernel at its distance and its number of revealed coordinates between (s
           held at the largest s read at a distance, linear in d between the distances read, zero
           from distance 16 on), each step at its own rate as above; the first-order reference
           charges only the pairs with none revealed between, at s = 0. The rules of fixed and
           midpoint_ties_spread, on the decoder's sets at 2 to 32 steps.

Writes OUT/NAME.csv.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os

import torch

from ..core import selection as SEL

PREFIX, SEQ = 192, 256                      # the decodes' windows: 64 decoded characters
BUDGETS = (2, 4, 8, 16, 32)
FIXED_RULES = ("contiguous", "spread", "separator", "midpoint", "random")
BETWEEN_RULES = ("contiguous", "random", "spread", "separator", "midpoint", "midpoint_ties_spread")
DECAY = 16                                  # the pair term is zero from this distance on
RANDOM_DRAWS = 16
CHANNEL_SEQ = 256                           # the channel pair's complete windows
CHANNEL_BUDGETS = (8, 16, 32, 64)


def globs(spec):
    out = []
    for g in spec.split():
        out += glob.glob(g)
    return sorted(set(out))


def rows_of(files):
    return [r for f in files for r in csv.DictReader(open(f))]


# ---- kernels ---------------------------------------------------------------------------------
def bracket(grid, d):
    """The read points either side of d and the weight on the upper one."""
    if d <= grid[0]:
        return grid[0], grid[0], 0.0
    if d >= grid[-1]:
        return grid[-1], grid[-1], 0.0
    lo = max(x for x in grid if x <= d)
    hi = min(x for x in grid if x >= d)
    return lo, hi, (0.0 if lo == hi else (d - lo) / (hi - lo))


def pair_term(K):
    grid = sorted(K)

    def kap(d):
        d = abs(d)
        if d >= DECAY:
            return 0.0
        lo, hi, f = bracket(grid, d)
        return K[lo] * (1 - f) + K[hi] * f
    return kap


def triplet_term(T3):
    grid = sorted({k[0] for k in T3})

    def tri(d1, d2):
        if d1 > DECAY or d2 > DECAY:
            return 0.0
        a0, a1, fa = bracket(grid, d1)
        b0, b1, fb = bracket(grid, d2)
        return ((1 - fa) * (1 - fb) * T3[(a0, b0)] + fa * (1 - fb) * T3[(a1, b0)]
                + (1 - fa) * fb * T3[(a0, b1)] + fa * fb * T3[(a1, b1)])
    return tri


def at(rate, r):
    return abs(float(r["rate"]) - rate) <= 1e-9


def triplet_table(trows, rate):
    """The triplet term by (d1, d2) at one rate, weighted by triples over the seeds."""
    T3 = collections.defaultdict(lambda: [0.0, 0])
    for r in trows:
        if at(rate, r):
            n = int(r["triples"])
            T3[(int(r["d1"]), int(r["d2"]))][0] += float(r["triplet"]) * n
            T3[(int(r["d1"]), int(r["d2"]))][1] += n
    return {k: s / n for k, (s, n) in T3.items()}


def crossing_fraction(trows, rate):
    """The fraction of spans of length d with a space in them (the triplet read's cross_bc at
    second gap d), the mean over the triplet rows at one rate."""
    agg = collections.defaultdict(lambda: [0.0, 0])
    for r in trows:
        if at(rate, r):
            agg[int(r["d2"])][0] += float(r["cross_bc"])
            agg[int(r["d2"])][1] += 1
    return {d: s / n for d, (s, n) in agg.items()}


def kernel(srows, trows, rate):
    """The kernel by distance at one rate: twice the pair read's gap per group, weighted by pairs
    over the seeds, the groups mixed at the corpus crossing fraction where both were read."""
    agg = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0]))
    for r in srows:
        if at(rate, r):
            n = int(r["pairs"])
            agg[int(r["d"])][r["group"]][0] += 2.0 * float(r["gap"]) * n
            agg[int(r["d"])][r["group"]][1] += n
    frac = crossing_fraction(trows, rate)
    out = {}
    for d, groups in agg.items():
        g = {k: s / n for k, (s, n) in groups.items()}
        if "within" in g and "crossing" in g and d in frac:
            out[d] = frac[d] * g["crossing"] + (1 - frac[d]) * g["within"]
        else:
            out[d] = g.get("crossing", g.get("within"))
    return out


def rates_read(rows):
    return {round(float(r["rate"]), 6) for r in rows}


def rate_tables(srows, trows, triplets=True):
    """(rate, pair term, triplet term) at every rate read in both inputs."""
    return [(rate, pair_term(kernel(srows, trows, rate)),
             triplet_term(triplet_table(trows, rate)) if triplets else None)
            for rate in sorted(rates_read(srows) & rates_read(trows))]


def at_rate(tables, r):
    """Both terms at mask rate r: linear between the rates read, the nearest outside them."""
    rates = [t[0] for t in tables]
    if r <= rates[0]:
        return tables[0][1], tables[0][2]
    if r >= rates[-1]:
        return tables[-1][1], tables[-1][2]
    i = max(j for j in range(len(rates)) if rates[j] <= r)
    if rates[i] == r:
        return tables[i][1], tables[i][2]
    (r0, k0, t0), (r1, k1, t1) = tables[i], tables[i + 1]
    w = (r - r0) / (r1 - r0)
    return ((lambda *d: (1 - w) * k0(*d) + w * k1(*d)),
            (lambda a, b: (1 - w) * t0(a, b) + w * t1(a, b)))


# ---- walks -----------------------------------------------------------------------------------
def walk(sets, tables, region):
    """Pair and triplet charges of one schedule (its sets in region coordinates), in bits over the
    region, each step at its own mask rate."""
    f1 = f2 = 0.0
    revealed = set()
    for S in sets:
        S = sorted(S)
        kap, tri = at_rate(tables, (region - len(revealed)) / region)
        cut = [any(x in revealed for x in range(S[j - 1] + 1, S[j])) for j in range(1, len(S))]
        for j in range(1, len(S)):
            if not cut[j - 1]:
                f1 += kap(S[j] - S[j - 1])
        for j in range(2, len(S)):
            if not cut[j - 2] and not cut[j - 1]:
                f2 += tri(S[j - 1] - S[j - 2], S[j] - S[j - 1])
        revealed |= set(S)
    return f1, f2


def decoder_sets(rule, steps, draw=0, device="cpu"):
    """The sets text8_decode's decode advances under a rule that reads only positions, in region
    coordinates, selected on ``device``; random draws from a CPU generator seeded with ``draw``."""
    if rule in ("midpoint", "midpoint_ties_spread"):
        ties = "spread" if rule == "midpoint_ties_spread" else "left"
        return [[p - PREFIX for p in s] for s in SEL.midpoint_schedule(SEQ, steps, PREFIX, ties=ties)]
    k = (SEQ - PREFIX) // steps
    mask = torch.ones(SEQ, dtype=torch.bool, device=device)
    mask[:PREFIX] = False
    gen = torch.Generator().manual_seed(draw)
    sets = []
    for t in range(steps):
        kk = k if t < steps - 1 else int(mask.sum())
        if rule == "contiguous":
            S = SEL.contiguous(mask, kk)
        elif rule == "spread":
            S = SEL.spread(mask, kk)
        elif rule == "random":
            S = SEL.random(mask, kk, generator=gen)
        elif rule == "separator":
            S = SEL.separator(mask, kk)
        else:
            raise ValueError(rule)
        S = sorted(int(p) for p in S.tolist())
        sets.append([p - PREFIX for p in S])
        mask[S] = False
        if not bool(mask.any()):
            break
    return sets


def between_kernel(srows):
    """rate -> {(d, s): kernel}: twice the gap of the reads made with the state between controlled,
    weighted by pairs over the seeds, the groups mixed at the read's crossing_fraction where both
    were read."""
    agg = collections.defaultdict(lambda: [0.0, 0])
    frac = {}
    for r in srows:
        n = int(r["pairs"])
        key = (round(float(r["rate"]), 6), int(r["d"]), int(r["between"]), r["group"])
        agg[key][0] += 2.0 * float(r["gap"]) * n
        agg[key][1] += n
        frac[int(r["d"])] = float(r["crossing_fraction"])
    g = {k: v / n for k, (v, n) in agg.items()}
    out = collections.defaultdict(dict)
    for rate, d, s, _ in g:
        w, c = g.get((rate, d, s, "within")), g.get((rate, d, s, "crossing"))
        out[rate][(d, s)] = (frac[d] * c + (1 - frac[d]) * w if w is not None and c is not None
                             else (c if c is not None else w))
    return dict(out)


def between_term(K):
    """The kernel at distance d with s revealed coordinates between, at one rate."""
    grid = sorted({d for d, _ in K})
    top = {d: max(s for dd, s in K if dd == d) for d in grid}

    def kap(d, s):
        d = abs(d)
        if d >= DECAY:
            return 0.0
        lo, hi, f = bracket(grid, d)
        return K[(lo, min(s, top[lo]))] * (1 - f) + K[(hi, min(s, top[hi]))] * f
    return kap


def between_walk(sets, tables, region):
    """(first-order reference, estimate with the state between) of one schedule, in bits over the
    region."""
    first = between = 0.0
    revealed = set()
    for S in sets:
        S = sorted(S)
        kap = at_rate(tables, (region - len(revealed)) / region)[0]
        for a, b in zip(S, S[1:]):
            s = sum(1 for x in range(a + 1, b) if x in revealed)
            between += kap(b - a, s)
            if s == 0:
                first += kap(b - a, 0)
        revealed |= set(S)
    return first, between


def estimate(schedules, tables, region):
    got = [walk(s, tables, region) for s in schedules]
    return (sum(g[0] for g in got) / len(got) / region, sum(g[1] for g in got) / len(got) / region)


def estimate_row(steps, rule, f1, f2, **extra):
    return dict(steps=steps, rule=rule, **extra, pairwise_bpc=f1, triplet_bpc=f2, second_order_bpc=f1 + f2)


def channel_sets(levels, steps, device="cpu", length=CHANNEL_SEQ):
    """The channel decode's sets (evaluation/text8_bpc.graded_path_bpc): per step, each advanced
    position with its level before the step."""
    lv = torch.zeros(length, dtype=torch.long, device=device)
    per = max(1, length * (levels - 1) // steps)
    out = []
    for _ in range(steps):
        pending = lv < levels - 1
        if not bool(pending.any()):
            break
        S = SEL.spread(pending, min(per, int(pending.sum()))).to(lv.device)
        out.append([(int(p), int(lv[p])) for p in S])
        lv[S] = (lv[S] + 1).clamp(max=levels - 1)
    return out


def channel_walk(sets, stage_tables, levels, length=CHANNEL_SEQ):
    """Pair charges of one channel schedule per stage, in bits over the window."""
    stages = [("commit", 0)] if levels == 2 else [("open", 0), ("commit", 1)]
    got = {name: 0.0 for name, _ in stages}
    done = set()
    for S in sets:
        rate = (length - len(done)) / length
        for name, from_level in stages:
            P = sorted(p for p, lv in S if lv == from_level)
            kap = at_rate(stage_tables[name], rate)[0]
            for a, b in zip(P, P[1:]):
                if not any(x in done for x in range(a + 1, b)):
                    got[name] += kap(b - a)
        for p, lv in S:
            if lv + 1 == levels - 1:
                done.add(p)
    return got


# ---- modes -----------------------------------------------------------------------------------
def inputs(args):
    srows, trows = rows_of(globs(args.structure)), rows_of(globs(args.triplet))
    if not srows or not trows:
        raise SystemExit("--structure and --triplet must each match at least one file")
    tables = rate_tables(srows, trows)
    if not tables:
        raise SystemExit("no mask rate is read in both the pair and the triplet files")
    return srows, trows, tables


def mode_fixed(args):
    srows, trows, tables = inputs(args)
    krows = [dict(rate=rate, d=d, kernel=v) for rate in sorted(rates_read(srows) & rates_read(trows))
             for d, v in sorted(kernel(srows, trows, rate).items())]
    write(krows, os.path.join(args.out, f"{args.name}_kernel.csv"))
    rows = []
    for steps in BUDGETS:
        for rule in FIXED_RULES:
            draws = range(RANDOM_DRAWS) if rule == "random" else (0,)
            schedules = [decoder_sets(rule, steps, s, device=args.sets_device) for s in draws]
            rows.append(estimate_row(steps, rule, *estimate(schedules, tables, SEQ - PREFIX)))
    return rows


def mode_decoded(args):
    _, _, tables = inputs(args)
    saved = torch.load(args.sets, weights_only=False)
    prefix, region = saved["prefix"], saved["seq"] - saved["prefix"]
    rows = []
    for (rule, steps, profile), sets in sorted(saved["sets"].items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if rule not in ("confidence", "min_distance") or profile is not None:
            continue
        windows = sets[0].shape[0]
        schedules = [[[int(p) - prefix for p in S[r].tolist()] for S in sets] for r in range(windows)]
        rows.append(estimate_row(steps, rule, *estimate(schedules, tables, region), windows=windows))
    return rows


def mode_channel(args):
    trows = rows_of(globs(args.triplet))
    tabs = {}
    for name, spec in (("absorbing", args.absorbing), ("open", args.class_kernel),
                       ("commit", args.character_kernel)):
        tabs[name] = rate_tables(rows_of(globs(spec)), trows, triplets=False)
        if not tabs[name]:
            raise SystemExit(f"no mask rate is read in both the {name} kernel and the triplet files")
    rows = []
    for steps in CHANNEL_BUDGETS:
        fa = channel_walk(channel_sets(2, steps, args.sets_device), {"commit": tabs["absorbing"]}, 2)
        g = channel_walk(channel_sets(3, steps, args.sets_device),
                         {"open": tabs["open"], "commit": tabs["commit"]}, 3)
        fo, fc = g["open"] / CHANNEL_SEQ, g["commit"] / CHANNEL_SEQ
        rows.append(dict(steps=steps, bound_absorbing_bpc=fa["commit"] / CHANNEL_SEQ,
                         bound_graded_bpc=fo + fc, bound_graded_class_bpc=fo,
                         bound_graded_character_bpc=fc))
    return rows


def mode_between(args):
    srows = rows_of(globs(args.structure))
    if not srows or "between" not in srows[0]:
        raise SystemExit("--structure must match pair reads made with text8_dependence kernel --between")
    K = between_kernel(srows)
    tables = [(rate, between_term(K[rate]), None) for rate in sorted(K)]
    region = SEQ - PREFIX
    rows = []
    for steps in BUDGETS:
        for rule in BETWEEN_RULES:
            draws = range(RANDOM_DRAWS) if rule == "random" else (0,)
            got = [between_walk(decoder_sets(rule, steps, s, device=args.sets_device), tables, region)
                   for s in draws]
            rows.append(dict(steps=steps, rule=rule,
                             first_order_reference_bpc=sum(g[0] for g in got) / len(got) / region,
                             state_between_bpc=sum(g[1] for g in got) / len(got) / region))
    return rows


MODES = {"fixed": mode_fixed, "decoded": mode_decoded, "channel": mode_channel, "between": mode_between}


def write(rows, path):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print("ROW " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()),
              flush=True)
    print(f"WROTE {path} ({len(rows)} rows)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=list(MODES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="", help="the stem of the output files (default: text8_<mode>)")
    ap.add_argument("--structure", default="", help="pair reads (text8_dependence kernel), globs")
    ap.add_argument("--triplet", default="", help="triplet reads (text8_dependence triplet), globs; every mode "
                    "but between")
    ap.add_argument("--sets", default="", help="decoded: the sets file text8_decode --save-sets wrote")
    ap.add_argument("--sets-device", default="cuda",
                    help="fixed, channel and between: the device the rules select on, the one the decodes ran on")
    ap.add_argument("--absorbing", default="", help="channel: pair reads of the absorbing model")
    ap.add_argument("--class-kernel", default="", help="channel: pair reads of the graded model's class stage")
    ap.add_argument("--character-kernel", default="",
                    help="channel: pair reads of the graded model's character stage")
    args = ap.parse_args(argv)
    args.name = args.name or f"text8_{args.mode}"
    os.makedirs(args.out, exist_ok=True)
    write(MODES[args.mode](args), os.path.join(args.out, f"{args.name}.csv"))


if __name__ == "__main__":
    main()
