"""The text8 displays: each reduces the reads its script made to the rows the paper shows.

    python -m lattice.experiments.tables <display> --results OUTPUT_ROOT/text8 --out DIR

``--results`` is the directory the text8 reads write to (scripts/_text8.sh), one subdirectory per
read: decodes/ (the decodes behind Table 3 (left), Table 18, Table 19 and Figure 5 (a), seeds 0-2),
chains/ (the same-schedule chains, seeds 0-15), kernel_85M/ and kernel_26M/ (the pair and triplet
reads of the 85M and 26M lattice models), estimate/ (the pairwise and second-order estimates, and
the estimates of Table 17), kernel_between/ (the pair reads with the state between the pair
controlled), ties_spread/ (the decode and chain of the midpoint rule with ties spread), sets/,
shapes/, overhead/, corners/, readable/, generation/, channels/, channel_kernels/ and
channel_estimate/. Means are over the seeds a read ran; a standard deviation is the sample
standard deviation over them (n - 1); an s.e.m. is that divided by the square root of the count.
Rows and columns carry the paper's labels (experiments/tables.py).
"""
from __future__ import annotations

import collections
import csv
import glob
import os

from ..utils.tables import write_rows


def _rows(path):
    return list(csv.DictReader(open(path)))


def _seed_rows(results, read, stem):
    """The rows of every seed file results/read/stem_s<seed>.csv, each tagged with its seed."""
    out = []
    for path in sorted(glob.glob(os.path.join(results, read, f"{stem}_s*.csv"))):
        seed = path.rsplit("_s", 1)[1][:-4]
        if not seed.isdigit():
            continue
        for r in _rows(path):
            r["seed"] = int(seed)
            out.append(r)
    if not out:
        raise SystemExit(f"no {stem}_s*.csv under {os.path.join(results, read)}: run the display's script")
    return out


def _mean(v):
    v = list(v)
    return sum(v) / len(v)


def _mean_sd(v):
    """Mean and sample standard deviation (n - 1; zero for one value)."""
    v = list(v)
    m = sum(v) / len(v)
    return m, (sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1)) ** 0.5


def _by(rows, *keys, value="bpc", where=None):
    """value lists grouped by the key columns (numbers parsed), rows filtered by ``where`` (a text
    value must match exactly, a number numerically)."""
    out = collections.defaultdict(list)
    for r in rows:
        if where and any(r.get(k) != v if isinstance(v, str) else abs(float(r[k]) - v) > 1e-9
                         for k, v in where.items()):
            continue
        if r.get(value, "") != "":
            out[tuple(_num(r[k]) for k in keys)].append(float(r[value]))
    return out


def _num(v):
    try:
        f = float(v)
        return int(f) if f == int(f) and "." not in v else f
    except (TypeError, ValueError):
        return v


# ---- decodes after a revealed prefix --------------------------------------------------------
# the rules of Table 3 (left), in its order; Figure 5 (a) draws the same six
METHODS = ("contiguous", "confidence", "random", "dilated", "spread", "separator")


def _decodes(args):
    return _by(_seed_rows(args.results, "decodes", "text8_decode"), "rule", "steps",
               where={"path": "decode"})


def table03_text8(args, rest):
    """Table 3 (left): bpc of the rules at 8, 16, 32 and 64 steps, mean over the decodes' seeds."""
    f = _decodes(args)
    rows = [dict(rule=rule, **{f"{T} steps": _mean(f[(rule, T)]) for T in (8, 16, 32, 64)})
            for rule in METHODS]
    write_rows(rows, args.out, "table03_text8")


def table18(args, rest):
    """Table 18: bpc at 64 to 2 steps for spread, random and contiguous: mean and sd over the seeds."""
    f = _decodes(args)
    rows = []
    for rule in ("spread", "random", "contiguous"):
        row = dict(rule=rule)
        for T in (64, 32, 16, 8, 4, 2):
            row[f"{T} steps"], row[f"{T} steps sd"] = _mean_sd(f[(rule, T)])
        rows.append(row)
    write_rows(rows, args.out, "table18")


def figure05_text8(args, rest):
    """Figure 5 (a): each rule's bpc at every budget and its excess over the exact left-to-right
    chain (contiguous at 64 steps), means over the seeds."""
    f = _decodes(args)
    chain = _mean(f[("contiguous", 64)])
    rows = [{"rule": rule, "steps": T, "bpc": _mean(f[(rule, T)]), "bpc above exact chain": _mean(f[(rule, T)]) - chain}
            for rule in METHODS for T in sorted(T for r, T in f if r == rule)]
    write_rows(rows, args.out, "figure05_text8")


# ---- same-schedule chains and the estimates -------------------------------------------------
def _chains(args):
    """(rule, steps) -> (decode bpc, chain bpc, seeds), means over the seeds each rule ran."""
    rows = _seed_rows(args.results, "chains", "text8_chains")
    dec = _by(rows, "rule", "steps", where={"path": "decode"})
    chn = _by(rows, "rule", "steps", where={"path": "schedule_chain"})
    return {k: (_mean(dec[k]), _mean(chn[k]), len(chn[k])) for k in chn}


def _estimates(args, name):
    return {(r["rule"], int(r["steps"])): r for r in _rows(os.path.join(args.results, "estimate", f"{name}.csv"))}


BUDGETS = (2, 4, 8, 16, 32)
# the rows of Tables 5, 13 and 15: the rules that read only positions, estimated on the decoder's sets
FIXED = ("contiguous", "random", "spread", "separator")


def _estimated_measured(args, name, column, label, rules=FIXED, row_label=None):
    """Rows (rule, steps, the estimate under ``label``, the measured excess of the decode over its
    own chain, the seeds the chain averages) at every budget."""
    ch, est = _chains(args), _estimates(args, name)
    rows = []
    for rule in rules:
        for T in BUDGETS:
            dec, chain, seeds = ch[(rule, T)]
            rows.append({"rule": row_label or rule, "steps": T, label: float(est[(rule, T)][column]),
                         "measured": dec - chain, "seeds": seeds})
    return rows


def table05(args, rest):
    """Table 5: the pairwise estimate on the 85M model's kernel against the measured excess over
    each schedule's own chain (random: the mean over 16 seeds), at every budget; the table shows 2,
    8 and 32 steps, its plot all of them."""
    write_rows(_estimated_measured(args, "fixed_85M", "pairwise_bpc", "estimate"), args.out, "table05")


def table13(args, rest):
    """Table 13: the pairwise estimate on the 26M model's kernel against the 85M model's measured
    excess."""
    write_rows(_estimated_measured(args, "fixed_26M", "pairwise_bpc", "predicted"), args.out, "table13")


def table15(args, rest):
    """Table 15: the second-order estimate (pairwise plus triplet) against the measured excess; for
    the confidence rule the estimate walks the sets its seed-0 decode selected."""
    rows = _estimated_measured(args, "fixed_85M", "second_order_bpc", "second-order estimate")
    rows += _estimated_measured(args, "decoded_85M", "second_order_bpc", "second-order estimate",
                                rules=("confidence",), row_label="confidence, measured after the decode")
    write_rows(rows, args.out, "table15")


def table14(args, rest):
    """Table 14: bpc of each rule's same-schedule chain (random: the mean over 16 seeds)."""
    ch = _chains(args)
    rows = [dict(rule=rule, **{f"{T} steps": ch[(rule, T)][1] for T in BUDGETS})
            for rule in ("contiguous", "confidence", "random", "spread", "separator")]
    write_rows(rows, args.out, "table14")


def table12(args, rest):
    """Table 12: the corpus-weighted kernel of the 85M model at the five mask rates, d = 1, 2, 4,
    8: the kernel the estimates read."""
    k = {(float(r["rate"]), int(r["d"])): float(r["kernel"])
         for r in _rows(os.path.join(args.results, "estimate", "fixed_85M_kernel.csv"))}
    rows = [{"mask rate": rate, **{f"d={d}": k[(rate, d)] for d in (1, 2, 4, 8)}}
            for rate in sorted({r for r, _ in k})]
    write_rows(rows, args.out, "table12")


def table16(args, rest):
    """Table 16: the pair kernel (twice the per-coordinate gap) of the 85M lattice model at mask
    rate 0.5 with s of the coordinates between the pair revealed and the rest of the window masked
    at the rate, d = 2, 3, 4 and s = 0 to d - 1, within a word and across a boundary, mean over the
    seeds."""
    g = _by(_seed_rows(args.results, "kernel_between", "kernel_between"), "group", "d", "between",
            value="gap", where={"rate": 0.5})
    rows = []
    for d in (2, 3, 4):
        for group, label in (("within", "within a word"), ("crossing", "across a boundary")):
            rows.append({"distance, pair": f"{d}, {label}",
                         **{f"s={s}": 2 * _mean(g[(group, d, s)]) for s in range(d) if (group, d, s) in g}})
    write_rows(rows, args.out, "table16")


# Table 17's rules: (the read's rule, the table's row label)
SEPARATION_RULES = (("separator", "separator"), ("midpoint", "midpoint"),
                    ("midpoint_ties_spread", "midpoint, ties spread"))


def table17(args, rest):
    """Table 17: the separator rule, the midpoint rule and the midpoint rule with ties spread at 8,
    16 and 32 steps: the first-order reference and the estimate with the state between
    (text8_estimate between, on the reads of Table 16), and the measured excess over each
    schedule's own chain (separator and midpoint: the chains of Table 14; midpoint with ties
    spread: its own decode and chain)."""
    est = _estimates(args, "between_85M")
    ch = _chains(args)
    ties = _seed_rows(args.results, "ties_spread", "text8_ties_spread")
    dec = _by(ties, "rule", "steps", where={"path": "decode"})
    chn = _by(ties, "rule", "steps", where={"path": "schedule_chain"})
    ch.update({k: (_mean(dec[k]), _mean(chn[k]), len(chn[k])) for k in chn})
    rows = []
    for rule, label in SEPARATION_RULES:
        for reading, value in (("first-order reference", lambda T: float(est[(rule, T)]["first_order_reference_bpc"])),
                               ("with the state between", lambda T: float(est[(rule, T)]["state_between_bpc"])),
                               ("measured", lambda T: ch[(rule, T)][0] - ch[(rule, T)][1])):
            rows.append(dict(rule=label, reading=reading, **{f"{T} steps": value(T) for T in (8, 16, 32)}))
    write_rows(rows, args.out, "table17")


# ---- kernels and total correlation ----------------------------------------------------------
# the pair groups of the kernel reads, with the labels of Table 10 and Figure 4 (a)
KERNEL_PAIRS = (("within", "inside one word"), ("crossing", "across a boundary"))


def _kernel_cells(args, rate):
    g = _by(_seed_rows(args.results, "kernel_85M", "kernel"), "group", "d", value="gap",
            where={"rate": rate})
    return {k: [2 * v for v in vals] for k, vals in g.items()}


def table10(args, rest):
    """Table 10: the pair kernel (twice the per-coordinate gap) at mask rate 0.5, inside one word
    and across a boundary, mean and sd over the seeds; a cell with too few pairs is left empty."""
    g = _kernel_cells(args, 0.5)
    rows = []
    for group, label in KERNEL_PAIRS:
        row = dict(pairs=label)
        for d in (1, 2, 4, 8, 16):
            if (group, d) in g:
                row[f"distance {d}"], row[f"distance {d} sd"] = _mean_sd(g[(group, d)])
        rows.append(row)
    write_rows(rows, args.out, "table10")


def figure04_text8(args, rest):
    """Figure 4 (a): the kernel at mask rate 0.5 by distance and pair group, mean and s.e.m. over
    the seeds."""
    g = _kernel_cells(args, 0.5)
    rows = []
    for group, label in KERNEL_PAIRS:
        for d in sorted(d for grp, d in g if grp == group):
            m, sd = _mean_sd(g[(group, d)])
            n = len(g[(group, d)])
            rows.append({"pairs": label, "distance": d, "kernel": m, "kernel s.e.m.": sd / n ** 0.5, "seeds": n})
    write_rows(rows, args.out, "figure04_text8")


def table11(args, rest):
    """Table 11: total correlation of the advanced set at mask rates 0.3, 0.5 and 0.7, 10^-3 bits
    per coordinate, mean and sd over the seeds."""
    g = _by(_seed_rows(args.results, "sets", "sets"), "rule", "rate", "k", value="gap")
    rows = []
    for rule in ("random", "spread"):
        for rate in (0.3, 0.5, 0.7):
            row = {"rule": rule, "mask rate": rate}
            for k in (8, 16, 32):
                m, sd = _mean_sd(g[(rule, rate, k)])
                row[f"width {k}"], row[f"width {k} sd"] = 1000 * m, 1000 * sd
            rows.append(row)
    write_rows(rows, args.out, "table11")


# ---- confidence, width profiles, selection time ---------------------------------------------
def table19(args, rest):
    """Table 19: at 16 steps, bpc and separation (means over the decodes' seeds) and the pairwise
    estimate on each rule's own sets (plain confidence and min-distance: the sets their seed-0
    decode selected; random and spread: the decoder's sets before the decode)."""
    fr = _seed_rows(args.results, "decodes", "text8_decode")
    bpc = _by(fr, "rule", "steps", where={"path": "decode"})
    sep = _by(fr, "rule", "steps", value="separation", where={"path": "decode"})
    fixed, decoded = _estimates(args, "fixed_85M"), _estimates(args, "decoded_85M")
    rows = []
    for rule, label in (("confidence", "confidence"), ("min_distance", "min-distance"), ("random", "random"),
                        ("spread", "spread")):
        est = (decoded if rule in ("confidence", "min_distance") else fixed)[(rule, 16)]
        rows.append({"selection rule": label, "bpc": _mean(bpc[(rule, 16)]), "separation": _mean(sep[(rule, 16)]),
                     "pairwise estimate": float(est["pairwise_bpc"])})
    write_rows(rows, args.out, "table19")


# Table 20's width profiles: (the read's profile, the table's row label)
PROFILES = (("uniform", "uniform"), ("wide_first", "wide first"), ("wide_late", "wide late"),
            ("geometric_wide_first", "geometric, wide first"), ("geometric_wide_late", "geometric, wide late"))


def table20(args, rest):
    """Table 20: complete decodes under width profiles, bpc for contiguous and spread at 8 and 16
    steps, and each profile's widths."""
    g = {(r["profile"], r["rule"], int(r["steps"])): (float(r["bpc"]), r["widths"])
         for r in _rows(os.path.join(args.results, "shapes", "text8_shapes_s0.csv"))}
    rows = []
    for profile, label in PROFILES:
        row = dict(profile=label)
        for rule in ("contiguous", "spread"):
            for T in (8, 16):
                row[f"{rule} {T}"] = g[(profile, rule, T)][0]
        row["widths, 8 steps"], row["widths, 16 steps"] = g[(profile, "spread", 8)][1], g[(profile, "spread", 16)][1]
        rows.append(row)
    write_rows(rows, args.out, "table20")


# Table 26's text8 rows: (the read's rule, the table's row label), in the table's order
OVERHEAD_RULES = (("contiguous", "contiguous"), ("random", "random"), ("confidence", "confidence"),
                  ("min_distance", "min-distance"), ("separator", "separator"), ("midpoint", "midpoint"),
                  ("spread", "spread"))


def table26_text8(args, rest):
    """Table 26, text8 rows: median forward-pass and selection time per step of the text8 decoder at
    16 steps, and the selection as a percentage of the forward pass."""
    read = {r["rule"]: r for r in _rows(os.path.join(args.results, "overhead", "text8_overhead_s0.csv"))}
    rows = []
    for rule, label in OVERHEAD_RULES:
        r = read[rule]
        rows.append({"rule": label, "forward, ms": float(r["forward_ms"]), "select, ms": float(r["select_ms"]),
                     "%": 100 * float(r["select_ms"]) / float(r["forward_ms"]),
                     "timed steps": int(r["timed_steps"]), "device": r["device"]})
    write_rows(rows, args.out, "table26_text8")


# ---- both corners, advances, free generation ------------------------------------------------
# the token block of Figure 2: (the read's name, the figure's row label)
CORNER_MODELS = (("causal", "Dedicated causal"), ("masked", "Dedicated masked"), ("lattice", "Lattice"),
                 ("two_stream", "Two-stream"))


def figure02_tokens(args, rest):
    """Figure 2, token block: each model's exact chain after 192 revealed characters (AR chain),
    and the masked ELBO of every model that can present a masked state."""
    rows = []
    for reading, label in (("exact chain", "AR chain"), ("masked ELBO", "Masked ELBO")):
        for name, model in CORNER_MODELS:
            for r in _rows(os.path.join(args.results, "corners", f"{name}_s0.csv")):
                if r["reading"] == reading:
                    rows.append(dict(reading=label, model=model, bpc=float(r["bpc"])))
    write_rows(rows, args.out, "figure02_tokens")


# Table 9's columns: (the read's advance, the table's column)
READABLE_POINTS = (("prefix, width one", "prefix, width 1"), ("interior, width one", "interior, width 1"),
                   ("interior, width eight", "interior, width 8"))


def table09(args, rest):
    """Table 9: mean and sd over the 16 draws of each architecture's loss at three advances; and,
    in table09_paired.csv, the paired excess of the two-stream over the bidirectional model at
    the interior width-one advance, which the caption quotes."""
    rows_in = _seed_rows(args.results, "readable", "text8_readable")
    g = _by(rows_in, "arch", "point")
    rows = []
    for arch in ("causal", "bidirectional", "two-stream"):
        row = dict(architecture=arch)
        for point, label in READABLE_POINTS:
            v = [x for x in g.get((arch, point), []) if x == x]
            if v:
                row[label], row[f"{label} sd"] = _mean_sd(v)
        rows.append(row)
    write_rows(rows, args.out, "table09")
    per = collections.defaultdict(dict)
    for r in rows_in:
        if r["point"] == "interior, width one":
            per[r["seed"]][r["arch"]] = float(r["bpc"])
    diff = [p["two-stream"] - p["bidirectional"] for p in per.values()]
    m, sd = _mean_sd(diff)
    write_rows([dict(comparison="two-stream minus bidirectional, interior width 1", mean=m, sd=sd,
                     positive=f"{sum(d > 0 for d in diff)} of {len(diff)}")], args.out, "table09_paired")


# Table 23's rules: (the read's rule, the table's row label)
FREEGEN_RULES = (("contiguous", "contiguous"), ("confidence", "confidence"), ("random", "random"),
                 ("spread", "spread"), ("word_aware_spread", "word-aware spread"))


def table23(args, rest):
    """Table 23: free generation, the causal scorer's bpc at 8, 16, 64 and 256 steps and the
    unigram KL divergence at 16 steps, mean and sd over the generation seeds."""
    rows_in = _rows(os.path.join(args.results, "generation", "text8_generation_s0.csv"))
    b = _by(rows_in, "rule", "steps")
    kl = _by(rows_in, "rule", "steps", value="unigram_kl")
    rows = []
    for rule, label in FREEGEN_RULES:
        row = dict(rule=label)
        for T in (8, 16, 64, 256):
            row[f"{T} steps"], row[f"{T} steps sd"] = _mean_sd(b[(rule, T)])
        row["unigram KL"], row["unigram KL sd"] = _mean_sd(kl[(rule, 16)])
        rows.append(row)
    write_rows(rows, args.out, "table23")


# ---- the channel pair -----------------------------------------------------------------------
def _channels(args):
    return {(r["channel"], int(r["steps"])): float(r["bpc"])
            for r in _rows(os.path.join(args.results, "channels", "text8_channels_s0.csv"))}


def table06(args, rest):
    """Table 6: complete-decode bpc of the absorbing model, of the graded model revealing classes
    then characters (both stages charged), and of the graded weights revealing each character at
    once."""
    c = _channels(args)
    rows = [dict(decode=label, **{f"{T} steps": c[(name, T)] for T in (8, 16, 32, 64)})
            for name, label in (("absorbing", "absorbing, spread"), ("graded", "graded, class then character"),
                                ("graded_character_at_once", "graded, character at once"))]
    write_rows(rows, args.out, "table06")


def table21(args, rest):
    """Table 21: each model's same-schedule chain, its excess (decode less chain) and its pairwise
    estimate."""
    c = _channels(args)
    b = {int(r["steps"]): r for r in _rows(os.path.join(args.results, "channel_estimate", "text8_channel.csv"))}
    cells = (("chain, absorbing", lambda T: c[("absorbing_chain", T)]),
             ("chain, graded", lambda T: c[("graded_chain", T)]),
             ("excess, absorbing", lambda T: c[("absorbing", T)] - c[("absorbing_chain", T)]),
             ("excess, graded", lambda T: c[("graded", T)] - c[("graded_chain", T)]),
             ("estimate, absorbing", lambda T: float(b[T]["bound_absorbing_bpc"])),
             ("estimate, graded (class + character)", lambda T: float(b[T]["bound_graded_bpc"])))
    rows = [dict(row=label, **{f"{T} steps": value(T) for T in (8, 16, 32, 64)}) for label, value in cells]
    write_rows(rows, args.out, "table21")


def table22(args, rest):
    """Table 22: pair kernels of the two models at mask rate 0.7 (twice the gap), within one word /
    across a boundary: the absorbing model's character kernel, the graded model's class kernel and
    its character kernel given the classes."""
    rows = []
    for name, label in (("absorbing", "absorbing, character"), ("graded_class", "graded, class"),
                        ("graded_character", "graded, character given classes")):
        g = {(r["group"], int(r["d"])): 2 * float(r["gap"])
             for r in _rows(os.path.join(args.results, "channel_kernels", f"{name}_s0.csv"))
             if abs(float(r["rate"]) - 0.7) < 1e-9}
        row = dict(kernel=label)
        for d in (1, 2, 4, 8):          # a cell with too few pairs is left empty
            row[f"distance {d}, within one word"] = g.get(("within", d), "")
            row[f"distance {d}, across a boundary"] = g.get(("crossing", d), "")
        rows.append(row)
    write_rows(rows, args.out, "table22")


DISPLAYS = {f.__name__: f for f in (
    figure02_tokens, table03_text8, figure04_text8, figure05_text8, table05, table06, table09, table10,
    table11, table12, table13, table14, table15, table16, table17, table18, table19, table20, table21,
    table22, table23, table26_text8)}
