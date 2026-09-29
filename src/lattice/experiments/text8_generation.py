"""Free generation under each selection rule, scored by the dedicated causal model (Table 23).

    python -m lattice.experiments.text8_generation --ckpt CKPT --scorer CKPT --data DATA/text8 \
        --out DIR [--steps 8,16,64,256] [--n 256] [--gseeds 3]

n sequences of 256 characters are generated from the all-mask state in T steps of 256/T
coordinates (the last step takes what is left): at every step the rule selects the positions and
the model's predicted law at each is sampled at temperature one. The finished text is scored by
the dedicated causal model (bits per character over characters 2 to 256) and compared with the
test split's unigram frequencies (KL divergence of the generated unigrams from the corpus's, in
bits).

Rules: contiguous, confidence (the maximum predicted probability), random, spread, and
word_aware_spread, the word-aware spread rule: the separator rule with words as the segments, at
most one position per word until every word has one, the words cut at the spaces already
generated and at the masked positions whose predicted probability of a space exceeds one half
(lattice.core.selection.spread_segments). At 256 steps every step advances one coordinate;
spread's single anchor is then the leftmost masked position, so spread and contiguous sample in
the same left-to-right order and differ only in their generation seeds. The generation seed of a
cell is 1000 T + 10 g + i, for generation seed g and the rule's index i in that list; the sampling
uses the global generator and the random rule a CPU generator, both seeded with it.

Writes OUT/text8_generation_s0.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from ..core import selection as SEL
from ..data.text8 import LN2, MASK, V, Text8
from ..evaluation.text8_bpc import forward_bf16, logits_at, select_sets
from ..models.text import load
from ..utils.results import Rows

RULES = ("contiguous", "confidence", "random", "spread", "word_aware_spread")
SPACE = 0


@torch.no_grad()
def generate(net, n, seq, steps, rule, seed, device):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    cur = torch.full((n, seq), MASK, dtype=torch.long, device=device)
    mask = torch.ones((n, seq), dtype=torch.bool, device=device)
    k = seq // steps
    for t in range(steps):
        kk = k if t < steps - 1 else int(mask[0].sum())
        if kk <= 0:
            break
        conf = None
        if rule == "confidence":
            conf = forward_bf16(net, cur).float().softmax(-1).max(-1).values
        if rule == "word_aware_spread":
            p_space = forward_bf16(net, cur).float().softmax(-1)[..., SPACE]
            seg = ((cur == SPACE) | (mask & (p_space > 0.5))).long().cumsum(1)
            S = torch.stack([SEL.spread_segments(mask[r], seg[r], kk) for r in range(n)]).to(device)
        else:
            S = select_sets(mask, kk, rule=rule, generator=g, conf=conf)
        pv = logits_at(net, cur, S).float().softmax(-1)
        tok = torch.multinomial(pv.reshape(-1, V), 1).reshape(n, -1)
        cur = cur.scatter(1, S, tok)
        mask = mask.scatter(1, S, False)
    return cur


@torch.no_grad()
def score_causal(scorer, xg, bs=64):
    """Bits per character of the text under the causal model, every character after the first."""
    total, cnt = 0.0, 0
    for i in range(0, xg.shape[0], bs):
        x = xg[i:i + bs]
        lg = forward_bf16(scorer, x).float()
        total += float(F.cross_entropy(lg[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1), reduction="sum"))
        cnt += x[:, 1:].numel()
    return total / cnt / LN2


def unigram_kl(xg, pdata):
    pg = np.bincount(xg.reshape(-1).cpu().numpy(), minlength=V).astype(float)
    pg /= pg.sum()
    keep = pg > 0
    return float((pg[keep] * np.log2(pg[keep] / pdata[keep])).sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="the model that generates (the lattice model)")
    ap.add_argument("--scorer", required=True, help="the dedicated causal model")
    ap.add_argument("--data", required=True, help="the directory of the text8 split (test.bin)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", default="8,16,64,256", help="step budgets, comma separated")
    ap.add_argument("--seq", type=int, default=256, help="sequence length")
    ap.add_argument("--n", type=int, default=256, help="sequences per cell")
    ap.add_argument("--gseeds", type=int, default=3, help="generation seeds per cell")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    net, meta = load(args.ckpt, kind="masked", device=args.device)
    scorer, _ = load(args.scorer, kind="causal", device=args.device)
    arr = np.asarray(Text8(args.data).test[:5_000_000])
    pdata = np.bincount(arr, minlength=V).astype(float)
    pdata /= pdata.sum()
    with Rows("text8_generation", args.out, meta=meta, seed=0) as rows:
        for steps in [int(s) for s in args.steps.split(",") if int(s) <= args.seq]:
            for i, rule in enumerate(RULES):
                for gs in range(args.gseeds):
                    xg = generate(net, args.n, args.seq, steps, rule, 1000 * steps + 10 * gs + i, args.device)
                    rows.add(steps=steps, rule=rule, gseed=gs, bpc=score_causal(scorer, xg),
                             unigram_kl=unigram_kl(xg, pdata))


if __name__ == "__main__":
    main()
