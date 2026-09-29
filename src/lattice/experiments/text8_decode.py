"""Teacher-forced decodes of held-out text8 under every selection rule, on one weight set.

    python -m lattice.experiments.text8_decode --ckpt CKPT --data DATA/text8 --out DIR \
        [--name NAME] [--rules R,...] [--steps 1,2,4,8,16,32,64] [--prefix 192] [--n 256] \
        [--seed 0] [--min-distance 8] [--separation] [--chain-within-step] [--save-sets] \
        [--profiles P,...] [--time-steps [--time-repeats 4] [--select-on-device]]

The protocol of Table 3 (left), Table 18 and Figure 5 (a): n windows of 256 test characters,
evenly spaced over the test split, the first 192 revealed and the last 64 decoded in T steps of
64/T coordinates (the last step takes what is left) chosen by the rule; every step is charged its
product law and its true characters are written back (evaluation/text8_bpc.py, path_bpc). One row
per rule and budget, in bits per decoded character. The random rule draws each window's sets from
a CPU generator seeded with seed + T; every other rule is deterministic, so its rows repeat at
every seed.

Rules: contiguous, random, spread, separator (at most one coordinate per masked run, each as far
as possible from the step's other picks), midpoint (the midpoint of each masked run),
midpoint_ties_spread (midpoint, with the picks among runs tied at the shortest length taken spread
evenly by position), dilated,
confidence, min_distance (confidence held to --min-distance between the picks of a step, topped up
by confidence when too few qualify); lattice.core.selection states each.

Variants
  --separation         the mean gap between consecutive selected positions of a step, averaged
                       over the steps (Table 19); empty when every step takes one position
  --chain-within-step  also each decode's same-schedule exact chain, rows with
                       path=schedule_chain: Table 14, and the measured excess (decode minus
                       chain) of Tables 5, 13, 15 and 17
  --save-sets          also every decode's selected sets, OUT/NAME_sets_s<seed>.pt, which
                       text8_estimate reads to estimate a rule that reads the model after the
                       decode
  --profiles           decodes under width profiles in place of equal widths (Table 20, with
                       --prefix 0: complete decodes of 256 characters): uniform; wide_first and
                       wide_late, widths proportional to (t+1)^-1 and (t+1)^+1;
                       geometric_wide_first and geometric_wide_late, widths scaled by 2/3 and 3/2
                       per step; integer widths of at least one summing to the decoded length
  --time-steps         the median time per step of the selection and of the call that charges
                       the step, over --time-repeats decodes of each cell (Table 26); the
                       confidence rules' extra call, which ranks the positions, is not timed;
                       --select-on-device selects through the rules' batch forms

Writes OUT/NAME_s<seed>.csv.
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np
import torch

from ..data.text8 import Text8, fixed_windows
from ..evaluation.text8_bpc import RULES, path_bpc
from ..models.text import load
from ..utils.results import Rows

# name -> (family, parameter); widths are proportional to (t + 1) ** p or to p ** t
PROFILES = {"uniform": ("power", 0.0), "wide_first": ("power", -1.0), "wide_late": ("power", 1.0),
            "geometric_wide_first": ("geometric", 2.0 / 3.0),
            "geometric_wide_late": ("geometric", 1.5)}


def profile_widths(length, steps, profile):
    """Integer widths of at least one, summing to ``length``, following the named profile."""
    family, p = PROFILES[profile]
    w = np.array([(t + 1.0) ** p if family == "power" else float(p) ** t for t in range(steps)])
    w = np.maximum(1, np.rint(w / w.sum() * length)).astype(int)
    while w.sum() > length:
        w[int(np.argmax(w))] -= 1
    while w.sum() < length:
        w[int(np.argmin(w))] += 1
    return [int(v) for v in w]


def device_name(device):
    return torch.cuda.get_device_name(0) if str(device).startswith("cuda") else "cpu"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="the model read (a lattice or dedicated masked model)")
    ap.add_argument("--data", required=True, help="the directory of the text8 split (test.bin)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="text8_decode", help="the stem of the output files")
    ap.add_argument("--rules", default="contiguous,random,spread,separator,dilated,confidence")
    ap.add_argument("--steps", default="1,2,4,8,16,32,64", help="step budgets, comma separated")
    ap.add_argument("--prefix", type=int, default=192, help="revealed characters in front of the decode")
    ap.add_argument("--seq", type=int, default=256, help="window length")
    ap.add_argument("--n", type=int, default=256, help="number of windows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-distance", type=int, default=8, help="min_distance's minimum distance")
    ap.add_argument("--separation", action="store_true")
    ap.add_argument("--chain-within-step", action="store_true")
    ap.add_argument("--save-sets", action="store_true")
    ap.add_argument("--profiles", default="", help="width profiles, comma separated (see above)")
    ap.add_argument("--time-steps", action="store_true")
    ap.add_argument("--time-repeats", type=int, default=4, help="decodes per cell whose steps are timed")
    ap.add_argument("--select-on-device", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    rules = args.rules.split(",")
    for rule in rules:
        if rule not in RULES:
            raise SystemExit(f"unknown rule {rule}; the rules are {', '.join(RULES)}")
    profiles = args.profiles.split(",") if args.profiles else []
    decoded = args.seq - args.prefix
    budgets = [int(s) for s in args.steps.split(",") if int(s) <= decoded]
    net, meta = load(args.ckpt, kind="masked", device=args.device)
    x = fixed_windows(Text8(args.data).test, args.n, args.seq).to(args.device)
    kept = {}
    with Rows(args.name, args.out, meta=meta, seed=args.seed) as rows:
        for steps in budgets:
            for rule in rules:
                for profile in profiles or [None]:
                    widths = profile_widths(decoded, steps, profile) if profile else None
                    common = dict(rule=rule, prefix=args.prefix, on_device=args.select_on_device,
                                  min_distance=args.min_distance if rule == "min_distance" else 0,
                                  **(dict(steps=steps) if profile in (None, "uniform") else dict(widths=widths)))
                    seeded = lambda: torch.Generator().manual_seed(args.seed + steps)
                    stats = {} if args.separation else None
                    timing = {} if args.time_steps else None
                    sets = [] if args.save_sets else None
                    bpc = path_bpc(net, x, generator=seeded(), stats=stats, timing=timing, sets=sets, **common)
                    for _ in range(args.time_repeats - 1 if timing is not None else 0):
                        path_bpc(net, x, generator=seeded(), timing=timing, **common)
                    row = dict(steps=steps, width=decoded // steps, rule=rule, prefix=args.prefix)
                    if profile:
                        row.update(profile=profile, widths=",".join(map(str, widths)))
                    key = dict(row)
                    row.update(path="decode", bpc=bpc)
                    if stats is not None and stats.get("gaps"):      # none when every step takes one
                        row["separation"] = sum(stats["gaps"]) / len(stats["gaps"])
                    if timing is not None:
                        row.update(forward_ms=1000 * statistics.median(timing["forward"]),
                                   select_ms=1000 * statistics.median(timing["select"]),
                                   timed_steps=len(timing["forward"]), device=device_name(args.device))
                    rows.add(**row)
                    if sets is not None:
                        kept[(rule, steps, profile)] = sets
                    if args.chain_within_step:
                        chain = path_bpc(net, x, generator=seeded(), chain_within=True, **common)
                        rows.add(**key, path="schedule_chain", bpc=chain)
    if args.save_sets:
        torch.save({"prefix": args.prefix, "seq": args.seq, "n": args.n, "seed": args.seed,
                    "min_distance": args.min_distance, "sets": kept},
                   Path(args.out) / f"{args.name}_sets_s{args.seed}.pt")

if __name__ == "__main__":
    main()
