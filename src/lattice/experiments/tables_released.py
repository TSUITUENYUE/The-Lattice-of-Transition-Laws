"""The displays of the experiments on released weights (LLaDA-8B-Base, MAR-B, SkyReels-V2).

Each display reads what its script wrote under --results (the scripts' $OUTPUT_ROOT/released):
    gsm8k/<rule>_s<steps>[_t<temperature>]/q<i>.json    GSM8K records (experiments/llada_gsm8k.py)
    llada_timing/<rule>_s64/q<i>.json                    the timed GSM8K decodes of Table 26
    genppl/seed<S>/...                                   samples and scores (experiments/llada_genppl.py)
    humaneval/<rule>_s<steps>/q<i>.json, scores/         (experiments/llada_humaneval.py)
    llada_kernel/kernel.csv                              (experiments/llada_kernel.py)
    mar/fid.csv, mar_kernel/, mar_kernel_info/           (experiments/mar_decode.py, mar_kernel.py)
    video/scores.csv, video/cells/*/run.json, video/kernel/kernel.csv   (experiments/video_*.py)
and writes its rows with the paper's labels (experiments/tables.py). Differences between rules are paired where the rules decode the same
problems: the paired standard error of a difference in accuracy is sqrt(((b + c)/n - d^2)/n) for
b and c problems solved by only one of the two rules and d = (b - c)/n, and the p-value is McNemar's
exact test on b and c. A binomial standard error is sqrt(p(1 - p)/n). Perplexity intervals are 95%
bootstrap intervals over samples (1,000 resamples, seed 0); a perplexity pools the summed
negative log-likelihoods and scored tokens of its samples.
"""
from __future__ import annotations

import collections
import csv
import itertools
import json
import math
import statistics as st
from pathlib import Path

import numpy as np
import torch

from ..models.skyreels import BASE_FRAMES, ROLLOUT_FRAMES, STEPS, segment_plan
from ..utils.tables import write_rows

# ---- LLaDA-8B-Base ------------------------------------------------------------------------------
GSM8K_N, HUMANEVAL_N = 1319, 164
BUDGETS = (8, 16, 32, 64, 128)
GENPPL_LENGTH, GENPPL_WIDTHS, GENPPL_SEEDS = 512, (64, 32, 16, 8), (0, 1, 2)
# the LLaDA rules as the paper's tables name them; LLaDA's own released rule is plain confidence
LABEL = {"random": "random", "confidence": "confidence", "confidence_min_distance": "min-distance",
         "confidence_separator": "separator constraint", "midpoint": "midpoint", "spread": "spread"}
CONSTRAINT = {"confidence_min_distance": "breaks_min_distance",
              "confidence_separator": "breaks_separator"}
# the comparisons of generative perplexity the text reports, (a, b): log perplexity of a minus b
GENPPL_COMPARISONS = (("confidence", "confidence_min_distance"), ("confidence", "random"),
                      ("random", "spread"), ("random", "midpoint"), ("spread", "midpoint"))


def _records(root, n):
    root = Path(root)
    missing = [i for i in range(n) if not (root / f"q{i:04d}.json").exists()]
    if missing:
        raise SystemExit(f"{root} lacks {len(missing)} of {n} records")
    return [json.loads((root / f"q{i:04d}.json").read_text()) for i in range(n)]


def _gsm8k(results, rule, steps, temperature=0.0):
    name = f"{rule}_s{steps}" + (f"_t{temperature:g}" if temperature > 0 else "")
    return _records(Path(results) / "gsm8k" / name, GSM8K_N)


def mcnemar(b, c):
    """Two-sided exact McNemar p-value for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def paired(a, b):
    """The paired comparison of outcomes a against outcomes b (lists of booleans), in percentage
    points."""
    n = len(a)
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    d = (only_a - only_b) / n
    return {"difference": 100 * d, "paired se": 100 * math.sqrt(max(0.0, (only_a + only_b) / n - d * d) / n),
            "solved by the first only": only_a, "solved by the second only": only_b,
            "McNemar p": mcnemar(only_a, only_b)}


def binomial(k, n):
    """k of n correct, in percent with its binomial standard error."""
    p = k / n
    return {"correct": k, "problems": n, "%": 100 * p, "binomial se": 100 * math.sqrt(p * (1 - p) / n)}


def relaxed_fraction(rule, records):
    """The fraction of a constrained rule's steps whose committed positions break its constraint."""
    key = CONSTRAINT.get(rule)
    if key is None:
        return None
    steps = [t for r in records for t in (r["steps"] if "steps" in r else r["trace"])]
    return sum(t[key] for t in steps) / len(steps)


def _repeated_4grams(ids):
    """The share of repeated four-token sequences in a sample's scorer tokens: one minus distinct over all."""
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return 1 - len(set(grams)) / len(grams)


def _genppl(results):
    """{(rule, width): [(seed, nll, targets, entropy, sample, repeated four-token share)]} over the three
    seeds."""
    root = Path(results) / "genppl"
    out = collections.defaultdict(list)
    for seed in GENPPL_SEEDS:
        protocol = json.loads((root / f"seed{seed}" / "protocol.json").read_text())
        for width in protocol["widths"]:
            for rule in protocol["rules"]:
                for i in range(protocol["samples"]):
                    path = root / f"seed{seed}" / "scores" / f"{rule}_w{width}" / f"s{i:04d}.json"
                    s = json.loads(path.read_text())
                    out[(rule, width)].append((seed, s["nll"], s["targets"], s["entropy_bits"], i,
                                               _repeated_4grams(s["scorer_token_ids"])))
    return out, root


def _pooled(cell):
    return math.exp(sum(x[1] for x in cell) / sum(x[2] for x in cell))


def _bootstrap_log_ppl(cell, generator, reps=1000):
    nll = torch.tensor([x[1] for x in cell], dtype=torch.float64)
    cnt = torch.tensor([x[2] for x in cell], dtype=torch.float64)
    idx = torch.randint(0, len(cell), (reps, len(cell)), generator=generator)
    return nll[idx].sum(1) / cnt[idx].sum(1)


def perplexity_interval(cell, reps=1000, seed=0):
    vals = _bootstrap_log_ppl(cell, torch.Generator().manual_seed(seed), reps).exp()
    return float(vals.quantile(0.025)), float(vals.quantile(0.975))


def log_ppl_difference(a, b, reps=1000, seed=0):
    """(log perplexity of a minus b, 95% interval), the two cells resampled independently."""
    g = torch.Generator().manual_seed(seed)
    diff = _bootstrap_log_ppl(a, g, reps) - _bootstrap_log_ppl(b, g, reps)
    return math.log(_pooled(a)) - math.log(_pooled(b)), float(diff.quantile(0.025)), float(diff.quantile(0.975))


def _genppl_relaxed(root, rule, width):
    traces = []
    for seed in GENPPL_SEEDS:
        for path in sorted((root / f"seed{seed}" / f"{rule}_w{width}").glob("b*.json")):
            traces += [dict(trace=s["trace"]) for s in json.loads(path.read_text())["samples"]]
    return relaxed_fraction(rule, traces)


def table04(args, rest):
    """Table 4: the GSM8K accuracy (greedy, percentage points, paired standard error) and the log
    generative perplexity (nats, 95% bootstrap interval) of the two constrained rules minus plain
    confidence, with the fraction of steps in which each relaxed its constraint; and, in
    table04_kernel.csv, the kernel's decay length at each mask rate, which sets the minimum
    distance."""
    rows = []
    for rule in ("confidence_min_distance", "confidence_separator"):
        for steps in BUDGETS:
            base, cell = _gsm8k(args.results, "confidence", steps), _gsm8k(args.results, rule, steps)
            p = paired([r["strict_match"] for r in cell], [r["strict_match"] for r in base])
            rows.append({"row": f"accuracy, {LABEL[rule]}", "steps": steps, "change": p.pop("difference"), **p,
                         "relaxed step fraction": relaxed_fraction(rule, cell)})
    cells, root = _genppl(args.results)
    for width in GENPPL_WIDTHS:
        diff, lo, hi = log_ppl_difference(cells[("confidence_min_distance", width)], cells[("confidence", width)])
        rows.append({"row": f"perplexity, {LABEL['confidence_min_distance']}", "steps": GENPPL_LENGTH // width,
                     "change": diff, "95% low": lo, "95% high": hi,
                     "relaxed step fraction": _genppl_relaxed(root, "confidence_min_distance", width)})
    write_rows(rows, args.out, "table04")
    write_rows(_kernel_rows(args.results), args.out, "table04_kernel")


def _kernel_rows(results):
    from .llada_kernel import decay_length
    profile = collections.defaultdict(dict)
    for r in csv.DictReader(open(Path(results) / "llada_kernel" / "kernel.csv")):
        profile[float(r["rate"])][int(r["distance"])] = float(r["kernel"])
    return [{"mask rate": rate, "kernel at distance 1": p.get(1), "decay length": decay_length(p)}
            for rate, p in sorted(profile.items())]


def table25(args, rest):
    """Table 25: generative perplexity of each rule at 8 to 64 steps pooled over three seeds, with
    its 95% interval and the seeds' own perplexities, the unigram entropy (bits, mean over samples)
    and the share of repeated four-token sequences (percent, mean over samples); and, in
    table25_comparisons.csv, the comparisons the text reports."""
    cells, root = _genppl(args.results)
    rows = []
    for rule in ("random", "confidence", "confidence_min_distance", "midpoint", "spread"):
        for width in GENPPL_WIDTHS:
            cell = cells[(rule, width)]
            lo, hi = perplexity_interval(cell)
            per_seed = [_pooled([x for x in cell if x[0] == s]) for s in GENPPL_SEEDS]
            rows.append({"rule": LABEL[rule], "steps": GENPPL_LENGTH // width, "samples": len(cell),
                         "generative perplexity": _pooled(cell), "95% low": lo, "95% high": hi,
                         "seed mean": st.mean(per_seed), "seed sd": st.stdev(per_seed),
                         "unigram entropy, bits": st.mean(x[3] for x in cell),
                         "repeated four-token sequences, %": 100 * st.mean(x[5] for x in cell),
                         "relaxed step fraction": _genppl_relaxed(root, rule, width)})
    write_rows(rows, args.out, "table25")
    comparisons = []
    for a, b in GENPPL_COMPARISONS:
        for width in GENPPL_WIDTHS:
            diff, lo, hi = log_ppl_difference(cells[(a, width)], cells[(b, width)])
            comparisons.append({"comparison": f"{LABEL[a]} minus {LABEL[b]}", "steps": GENPPL_LENGTH // width,
                                "log perplexity difference, nats": diff, "95% low": lo, "95% high": hi,
                                "resolved": "tie" if lo <= 0 <= hi else ("first higher" if diff > 0 else "second higher")})
    write_rows(comparisons, args.out, "table25_comparisons")


def repetition_rows(cells):
    """For each number of steps in cells, {steps: (confidence, minimum distance)} with one row per sample
    (nll, scored tokens, unigram entropy, repeated four-token share): the minimum-distance rule's log
    perplexity minus plain confidence's, pooled; the rule's coefficient when each sample's log perplexity
    is regressed on the rule, its unigram entropy and its repeated four-token share, with a 95% bootstrap
    interval (4,000 resamples, samples redrawn within each rule, one generator at seed 0 over the steps in
    the order given); and the pooled difference among the samples whose repeated four-token share lies
    below the median of both rules' samples."""
    rng = np.random.default_rng(0)
    rows = []
    for steps, (c, m) in cells.items():
        X = np.vstack([c, m])
        y = X[:, 0] / X[:, 1]
        z = np.r_[np.zeros(len(c)), np.ones(len(m))]

        def coef(i):
            A = np.c_[np.ones(len(i)), z[i], X[i, 2], X[i, 3]]
            return np.linalg.lstsq(A, y[i], rcond=None)[0][1]

        boot = np.array([coef(np.r_[rng.integers(0, len(c), len(c)), len(c) + rng.integers(0, len(m), len(m))])
                         for _ in range(4000)])
        median = np.median(X[:, 3])
        cl, ml = c[c[:, 3] < median], m[m[:, 3] < median]
        pooled = lambda v: v[:, 0].sum() / v[:, 1].sum()  # noqa: E731
        rows.append(dict(steps=steps, pooled=float(pooled(m) - pooled(c)), controlled=float(coef(np.arange(len(X)))),
                         ci95_lo=float(np.percentile(boot, 2.5)), ci95_hi=float(np.percentile(boot, 97.5)),
                         less_repetitive_half=float(pooled(ml) - pooled(cl)),
                         samples_below_median=f"{len(cl)} confidence, {len(ml)} min-distance"))
    return rows


def table24(args, rest):
    """Table 24: the minimum-distance rule against plain confidence on the samples of Table 25, at 8
    to 64 steps, with the share of repeated four-token sequences held fixed (repetition_rows): the
    table's rows (pooled, controlled with its 95% interval, less repetitive half), each at every
    number of steps."""
    cells, _ = _genppl(args.results)
    by_steps = {GENPPL_LENGTH // width: tuple(np.array([(x[1], x[2], x[3], x[5]) for x in cells[(rule, width)]],
                                                       dtype=float)
                                              for rule in ("confidence", "confidence_min_distance"))
                for width in GENPPL_WIDTHS}
    got = repetition_rows(by_steps)
    rows = [{"row": "pooled", "steps": g["steps"], "change": g["pooled"]} for g in got]
    rows += [{"row": "controlled", "steps": g["steps"], "change": g["controlled"], "95% low": g["ci95_lo"],
              "95% high": g["ci95_hi"]} for g in got]
    rows += [{"row": "less repetitive half", "steps": g["steps"], "change": g["less_repetitive_half"],
              "samples below the median": g["samples_below_median"]} for g in got]
    write_rows(rows, args.out, "table24")


def _humaneval(results, rule, steps):
    root = Path(results) / "humaneval" / f"{rule}_s{steps}"
    return [r["status"] == "passed" for r in _records(root / "scores", HUMANEVAL_N)], _records(root, HUMANEVAL_N)


def table27(args, rest):
    """Table 27: HumanEval pass@1 of plain confidence and its two constrained forms, and GSM8K under
    sampling at 64 steps (table27.csv); their paired comparisons, which the text and the caption
    quote (table27_paired.csv); and plain confidence on GSM8K at 256 steps, the authors' setting
    reproduced, which the caption quotes (table27_gsm8k_256.csv)."""
    rules = ("confidence", "confidence_min_distance", "confidence_separator")
    rows, passed = [], {}
    for rule in rules:
        for steps in BUDGETS:
            passed[(rule, steps)], generated = _humaneval(args.results, rule, steps)
            rows.append({"block": "HumanEval pass@1", "rule": LABEL[rule], "steps": steps,
                         **binomial(sum(passed[(rule, steps)]), HUMANEVAL_N),
                         "relaxed step fraction": relaxed_fraction(rule, generated)})
    sampled = {r: _gsm8k(args.results, r, 64, temperature=1.0) for r in ("confidence", "confidence_min_distance")}
    for rule, cell in sampled.items():
        rows.append({"block": "GSM8K, sampling, 64 steps", "rule": LABEL[rule], "steps": 64,
                     **binomial(sum(r["strict_match"] for r in cell), GSM8K_N),
                     "relaxed step fraction": relaxed_fraction(rule, cell)})
    write_rows(rows, args.out, "table27")
    comparisons = []
    for a, b in (("confidence_min_distance", "confidence"), ("confidence_separator", "confidence"),
                 ("confidence_separator", "confidence_min_distance")):
        for steps in BUDGETS:
            comparisons.append({"block": "HumanEval pass@1", "comparison": f"{LABEL[a]} against {LABEL[b]}",
                                "steps": steps, **paired(passed[(a, steps)], passed[(b, steps)])})
    comparisons.append({"block": "GSM8K, sampling, 64 steps",
                        "comparison": f"{LABEL['confidence_min_distance']} against {LABEL['confidence']}", "steps": 64,
                        **paired([r["strict_match"] for r in sampled["confidence_min_distance"]],
                                 [r["strict_match"] for r in sampled["confidence"]])})
    write_rows(comparisons, args.out, "table27_paired")
    full = _gsm8k(args.results, "confidence", 256)
    write_rows([{"block": "GSM8K, greedy, 256 steps", "rule": LABEL["confidence"], "steps": 256,
                 **binomial(sum(r["strict_match"] for r in full), GSM8K_N),
                 "flexible match, %": 100 * sum(r["flexible_match"] for r in full) / GSM8K_N}],
               args.out, "table27_gsm8k_256")


TIMED_RULES = ("random", "confidence", "confidence_min_distance", "confidence_separator", "midpoint", "spread")


def table26_llada(args, rest):
    """Table 26, LLaDA rows: the median time of a decoding step's forward pass and of its selection
    (for the confidence rules the probability of the proposed tokens included) on the first 20
    GSM8K problems at 64 steps, the selection as a percentage of the forward pass, and the median
    time of the argmax or draw (score)."""
    rows = []
    for rule in TIMED_RULES:
        steps = [t for r in _records(Path(args.results) / "llada_timing" / f"{rule}_s64", 20) for t in r["steps"]]
        forward = 1000 * st.median(t["forward_seconds"] for t in steps)
        select = 1000 * st.median(t["select_seconds"] for t in steps)
        rows.append({"rule": LABEL[rule], "forward, ms": forward, "select, ms": select, "%": 100 * select / forward,
                     "score, ms": 1000 * st.median(t["score_seconds"] for t in steps), "timed steps": len(steps),
                     "model": "LLaDA-8B-Base", "steps": 64})
    write_rows(rows, args.out, "table26_llada")


# ---- MAR-B ---------------------------------------------------------------------------------------
MAR_STEPS = (8, 16, 32, 64, 256)
# the rows of Table 3 (right), in its order, and the steps each is read at; Figure 5 (b) draws the first five
MAR_CELLS = {"raster": MAR_STEPS, "random": MAR_STEPS, "midpoint": MAR_STEPS, "spread": MAR_STEPS,
             "low_discrepancy": MAR_STEPS, "nested": (8, 16, 32), "spread_raster": (8, 16)}
MAR_RULES = tuple(MAR_CELLS)
MAR_FIGURE_RULES = ("raster", "random", "midpoint", "spread", "low_discrepancy")
MAR_LABEL = {"raster": "raster", "random": "random, the model's own", "midpoint": "midpoint", "spread": "spread",
             "low_discrepancy": "low-discrepancy", "nested": "nested", "spread_raster": "spread, then raster"}
MAR_SEEDS = {("random", 8): (0, 1, 2), ("random", 16): (0, 1, 2),       # cells the paper averages over seeds
             ("midpoint", 8): (0, 1, 2), ("midpoint", 16): (0, 1, 2)}


def _mar(results):
    """(rule, steps) -> FID-50K and Inception Score at guidance 2.9: the mean over the seeds the paper averages
    (MAR_SEEDS), else seed 0; a missing cell stops the display."""
    rows = {}
    for r in csv.DictReader(open(Path(results) / "mar" / "fid.csv")):
        if float(r["cfg"]) == 2.9:
            rows[(r["rule"], int(r["num_iter"]), int(r["seed"]))] = r
    out = {}
    for rule, steps in ((rule, s) for rule in MAR_RULES for s in MAR_CELLS[rule]):
        seeds = MAR_SEEDS.get((rule, steps), (0,))
        missing = [k for k in seeds if (rule, steps, k) not in rows]
        if missing:
            raise SystemExit(f"Table 3 lacks {rule} at {steps} steps, seed {missing}: run the display's script")
        cells = [rows[(rule, steps, k)] for k in seeds]
        out[(rule, steps)] = dict(fid=st.mean(float(c["fid"]) for c in cells),
                                  isc=st.mean(float(c["isc"]) for c in cells), n=int(cells[0]["n"]), seeds=len(seeds))
    return out


def table03_mar(args, rest):
    """Table 3 (right): FID-50K of MAR-B under seven reveal orders at the steps each is read at, with the
    Inception Score; random and midpoint at 8 and 16 steps are the mean of three sampling seeds."""
    cells = _mar(args.results)
    rows = [{"rule": MAR_LABEL[rule], "steps": steps, "FID": cells[(rule, steps)]["fid"],
             "Inception Score": cells[(rule, steps)]["isc"], "images": cells[(rule, steps)]["n"],
             "seeds": cells[(rule, steps)]["seeds"]}
            for rule in MAR_RULES for steps in MAR_CELLS[rule]]
    write_rows(rows, args.out, "table03_mar")


def figure05_mar(args, rest):
    """Figure 5 (b): FID-50K of five reveal orders above our read of MAR-B's own random order at 256
    steps (the cells of Table 3; the published score is 2.31)."""
    cells = _mar(args.results)
    ref = cells[("random", 256)]["fid"]
    rows = [{"panel": "b", "rule": MAR_LABEL[rule], "steps": steps, "FID": cells[(rule, steps)]["fid"],
             "FID above random at 256": cells[(rule, steps)]["fid"] - ref}
            for rule in MAR_FIGURE_RULES for steps in MAR_CELLS[rule]]
    write_rows(rows, args.out, "figure05_mar")


def figure04_mar(args, rest):
    """Figure 4 (b): the mean relative displacement of the decoder output at each grid distance and
    revealed fraction, and that divided by its value at distance 1 (the figure shows fraction 0.5)."""
    agg = collections.defaultdict(list)
    for r in csv.DictReader(open(Path(args.results) / "mar_kernel" / "kernel.csv")):
        agg[(float(r["fraction"]), int(r["distance"]))].append(float(r["rel_displacement"]))
    rows = []
    for frac in sorted({f for f, _ in agg}):
        one = st.mean(agg[(frac, 1)])
        for d in sorted(d for f, d in agg if f == frac):
            mean = st.mean(agg[(frac, d)])
            rows.append({"panel": "b", "revealed fraction": frac, "distance": d, "relative displacement": mean,
                         "kernel / kernel at 1": mean / one, "readings": len(agg[(frac, d)])})
    write_rows(rows, args.out, "figure04_mar")


# ---- SkyReels-V2 -----------------------------------------------------------------------------------
# Table 28's rows, the shipped depth first: (memory depth in frames, the table's label)
VIDEO_DEPTHS = ((17, "17 frames, as shipped"), (5, "5 frames"), (25, "25 frames"), (33, "33 frames"),
                (57, "57 frames"))
SHIPPED = 17
# the four VBench dimensions declared before decoding: (the scores' name, the table's column)
VIDEO_DIMENSIONS = (("subject_consistency", "subject consistency"), ("background_consistency", "background consistency"),
                    ("motion_smoothness", "motion smoothness"), ("imaging_quality", "imaging quality"))


def _video_scores(root):
    """(dimension, overlap) -> {prompt: score}, from the scoring of the rollouts."""
    score = collections.defaultdict(dict)
    for r in csv.DictReader(open(Path(root) / "scores.csv")):
        score[(r["dimension"], int(r["overlap"]))][int(r["prompt"])] = float(r["score"])
    return score


def video_steps(overlap):
    """The decoding steps of a rollout at one memory depth, from its segment plan: 30 per segment."""
    return STEPS * len(segment_plan(ROLLOUT_FRAMES, BASE_FRAMES, overlap))


def _frame_kernel_rows(root):
    """The kernel between frames by distance in latent frames: at each probed step of the trajectory and
    for each metric, the mean over prompts of each prompt's mean over moved and other frames, and its
    value divided by its value at distance 1 (mean and standard error over prompts)."""
    kernel = list(csv.DictReader(open(Path(root) / "kernel" / "kernel.csv")))
    rows = []
    for step in sorted({int(r["step"]) for r in kernel}):
        for metric in ("relative_flow_mse", "relative_clean_mse"):
            per_prompt = collections.defaultdict(lambda: collections.defaultdict(list))
            for r in kernel:
                if int(r["step"]) == step:
                    per_prompt[int(r["prompt"])][int(r["distance"])].append(float(r[metric]))
            means = {p: {d: st.mean(v) for d, v in by_d.items()} for p, by_d in per_prompt.items()}
            sigma = next(float(r["sigma"]) for r in kernel if int(r["step"]) == step)
            for d in sorted(means[0]):
                ratios = [means[p][d] / means[p][1] for p in sorted(means)]
                rows.append({"step index": step, "sigma": sigma, "metric": metric, "distance": d,
                             "kernel": st.mean(means[p][d] for p in means), "relative to distance 1": st.mean(ratios),
                             "relative to distance 1 se": st.stdev(ratios) / math.sqrt(len(ratios))})
    return rows


def table28(args, rest):
    """Table 28: each memory depth's decoding steps, from its segment plan (checked against the steps
    every rollout at that depth records), and the four VBench dimensions declared before decoding over
    the whole rollout: at the shipped depth the mean of the three prompts, at every other depth the mean
    of the three paired differences from the shipped depth with their standard deviation over sqrt 3 (the
    paired standard error); and, in table28_kernel.csv, the kernel between frames that Appendix D quotes
    beside the table (_frame_kernel_rows)."""
    root = Path(args.results) / "video"
    score = _video_scores(root)
    rows = []
    for overlap, label in VIDEO_DEPTHS:
        steps = video_steps(overlap)
        recorded = {json.loads(p.read_text())["decoding_steps"] for p in (root / "cells").glob(f"p*_ov{overlap}_s0/run.json")}
        if recorded != {steps}:
            raise SystemExit(f"the rollouts at depth {overlap} record {sorted(recorded)} decoding steps; "
                             f"the segment plan gives {steps}")
        row = {"memory depth": label, "steps": steps}
        for dim, name in VIDEO_DIMENSIONS:
            own, shipped = score[(dim, overlap)], score[(dim, SHIPPED)]
            if overlap == SHIPPED:
                row[name], row[f"{name} se"] = st.mean(own.values()), ""
            else:
                diffs = [own[p] - shipped[p] for p in sorted(own)]
                row[name], row[f"{name} se"] = st.mean(diffs), st.stdev(diffs) / math.sqrt(len(diffs))
        rows.append(row)
    write_rows(rows, args.out, "table28")
    write_rows(_frame_kernel_rows(root), args.out, "table28_kernel")


def _kernel_at_half(path, column):
    """{distance: mean over the file's rows} at revealed fraction one half."""
    agg = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if abs(float(r["fraction"]) - 0.5) < 1e-9:
            agg[int(r["distance"])].append(float(r[column]))
    return {d: st.mean(v) for d, v in agg.items()}


def _floors(path):
    return {(r["rule"], int(r["steps"])): float(r["floor"]) for r in csv.DictReader(open(path))}


# the rows of Table 2 (right), in its order: by the displacement floor at 8 steps, highest first
MARINFO_RULES = ("raster", "spread_raster", "random", "midpoint", "low_discrepancy", "nested", "spread")


def table02(args, rest):
    """Table 2: the displacement and information kernels of MAR-B at revealed fraction one half, distances 1
    to 5 (table02_kernels.csv, the left half); every rule's 8-step first-order floor under each kernel beside
    its FID-50K from Table 3 (table02.csv, the right half); and the checks the text states
    (table02_checks.csv): each floor keeps its 8-step rule order at 16, 32 and 64 steps; the pairs the two
    kernels order oppositely, with their relative gap on the displacement floor; midpoint's relative gap
    below random on each floor."""
    root = Path(args.results)
    disp = _kernel_at_half(root / "mar_kernel" / "kernel.csv", "rel_displacement")
    info = _kernel_at_half(root / "mar_kernel_info" / "information.csv", "mi_bits")
    write_rows([{"d": d, "displacement": disp[d], "information": info[d]} for d in range(1, 6)],
               args.out, "table02_kernels")
    floor = {"displacement": _floors(root / "mar_kernel" / "floors.csv"),
             "information": _floors(root / "mar_kernel_info" / "floors.csv")}
    fid = _mar(args.results)
    write_rows([{"rule": MAR_LABEL[r], "displacement floor": floor["displacement"][(r, 8)],
                 "information floor": floor["information"][(r, 8)], "FID": fid[(r, 8)]["fid"]}
                for r in MARINFO_RULES], args.out, "table02")
    checks = []
    for name, f in floor.items():
        at8 = sorted(MAR_RULES, key=lambda r: f[(r, 8)])
        for t in (16, 32, 64):
            checks.append(dict(check=f"the {name} floor keeps its 8-step order at {t} steps",
                               value=sorted(MAR_RULES, key=lambda r: f[(r, t)]) == at8))
    fd, fi = floor["displacement"], floor["information"]
    for a, b in itertools.combinations(MAR_RULES, 2):
        if (fd[(a, 8)] - fd[(b, 8)]) * (fi[(a, 8)] - fi[(b, 8)]) < 0:
            checks.append(dict(check=f"{MAR_LABEL[a]} and {MAR_LABEL[b]} ordered oppositely: gap on the displacement floor",
                               value=abs(fd[(a, 8)] - fd[(b, 8)]) / max(fd[(a, 8)], fd[(b, 8)])))
    for name, f in floor.items():
        checks.append(dict(check=f"midpoint below random on the {name} floor",
                           value=(f[("random", 8)] - f[("midpoint", 8)]) / f[("random", 8)]))
    write_rows(checks, args.out, "table02_checks")


DISPLAYS = {f.__name__: f for f in (table02, table03_mar, figure04_mar, table04, figure05_mar, table24, table25,
                                    table26_llada, table27, table28)}
