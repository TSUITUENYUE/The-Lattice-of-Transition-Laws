"""Complete decodes on the two channels of the channel pair (Tables 6 and 21).

    python -m lattice.experiments.text8_channels --absorbing CKPT --graded CKPT \
        --data DATA/text8 --out DIR [--steps 8,16,32,64] [--n 512] [--chain-within-step]

n windows of 256 test characters, evenly spaced over the test split, are decoded from the
all-mask state in T steps by the spread rule and read in bits per character
(evaluation/text8_bpc.py, graded_path_bpc). The absorbing model commits 256/T characters a step.
The graded model advances 512/T coordinates a step by one level, so a step first reveals a set's
classes (space, vowel, consonant) and the next commits the same set's characters; both stages are
charged, the class at the model's probability of the true class. The graded weights are also read
revealing each character at once, 256/T characters committed a step and the class level unused
(graded_character_at_once, the row "graded, character at once" of Table 6). Both models are
deterministic on these windows.

With --chain-within-step each model's same-schedule exact chain is read too (rows absorbing_chain
and graded_chain): every step's set revealed one coordinate at a time, in position order, each
charged at its stage from the state holding the coordinates before it. A decode less its chain is
the excess of Table 21.

Writes OUT/text8_channels_s0.csv.
"""
from __future__ import annotations

import argparse

from ..data.text8 import Graded, Text8, fixed_windows
from ..evaluation.text8_bpc import graded_path_bpc
from ..models.text import load
from ..utils.results import Rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--absorbing", required=True, help="the absorbing model of the channel pair")
    ap.add_argument("--graded", required=True, help="the graded model of the channel pair")
    ap.add_argument("--data", required=True, help="the directory of the text8 split (test.bin)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", default="8,16,32,64", help="step budgets, comma separated")
    ap.add_argument("--seq", type=int, default=256, help="window length")
    ap.add_argument("--n", type=int, default=512, help="number of windows")
    ap.add_argument("--chain-within-step", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    x = fixed_windows(Text8(args.data).test, args.n, args.seq).to(args.device)
    models = []
    for name, path, channel in (("absorbing", args.absorbing, Graded(levels=2)),
                                ("graded", args.graded, Graded(levels=3))):
        net, meta = load(path, kind="masked", device=args.device)
        models.append((name, net, channel, meta))
    meta = {f"{n}_step": m.get("step") for n, _, _, m in models}
    with Rows("text8_channels", args.out, meta=meta, seed=0) as rows:
        for steps in [int(s) for s in args.steps.split(",")]:
            for name, net, channel, _ in models:
                common = dict(levels=channel.levels, steps=steps, vocab=channel.vocab)
                rows.add(channel=name, **common, bpc=graded_path_bpc(net, channel, x, steps))
                if name == "graded":
                    rows.add(channel="graded_character_at_once", **common,
                             bpc=graded_path_bpc(net, channel, x, steps, character_at_once=True))
                if args.chain_within_step:
                    rows.add(channel=f"{name}_chain", **common,
                             bpc=graded_path_bpc(net, channel, x, steps, chain_within=True))


if __name__ == "__main__":
    main()
