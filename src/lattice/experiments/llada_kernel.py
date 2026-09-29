"""The pair kernel of LLaDA-8B-Base on held-out text, read before any decoding.

    python -m lattice.experiments.llada_kernel --model DATA/llada/LLaDA-8B-Base \
        --corpus DATA/llada/wikitext103_test.txt --out DIR

A window of 256 corpus tokens is masked at a rate; the entropy of every masked position is read;
one masked position is revealed to its true token and the decrease in entropy at every other
masked position is recorded against its distance to the revealed one. 16 windows per rate, 8
revealed positions per window, rates 0.3, 0.5 and 0.7. The decay length of a rate is the first
distance at which the mean decrease falls to a tenth of its value at distance 1; it sets the
minimum distance of the min-distance rule (6 on the released weights). Writes OUT/kernel.csv
(rate, distance, mean decrease in bits, pairs) and prints each rate's decay length.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..models.llada import load_llada, logits
from ..utils.results import Rows

LN2 = 0.6931471805599453


@torch.no_grad()
def entropies(model, x):
    """Entropy in bits of the model's prediction at every position of x."""
    p = logits(model, x).float().softmax(-1)
    return -(p * (p + 1e-9).log()).sum(-1) / LN2


def decay_length(profile):
    """The first distance whose kernel is at most a tenth of the kernel at distance 1."""
    one = profile.get(1, 0.0)
    if one <= 0:
        return None
    return next((d for d in sorted(profile) if profile[d] <= 0.1 * one), None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the LLaDA-8B-Base directory")
    ap.add_argument("--corpus", required=True, help="a text file of held-out natural text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--length", type=int, default=256, help="tokens per window")
    ap.add_argument("--rates", default="0.3,0.5,0.7", help="mask rates")
    ap.add_argument("--n", type=int, default=16, help="windows per rate")
    ap.add_argument("--probes", type=int, default=8, help="revealed positions per window")
    ap.add_argument("--dmax", type=int, default=64, help="distances beyond this are pooled into it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    dev = args.device
    model, tokenizer, mask_id = load_llada(args.model, dev)
    ids = tokenizer(Path(args.corpus).read_text(errors="ignore"), return_tensors="pt").input_ids[0]
    print(f"corpus of {len(ids)} tokens, mask id {mask_id}", flush=True)
    g = torch.Generator().manual_seed(args.seed)
    L, D = args.length, args.dmax
    rows = Rows("kernel", args.out)
    for rate in [float(v) for v in args.rates.split(",")]:
        tot = torch.zeros(D + 1, device=dev)
        cnt = torch.zeros(D + 1, device=dev)
        for _ in range(args.n):
            s = int(torch.randint(0, max(1, len(ids) - L), (1,), generator=g))
            x0 = ids[s:s + L].to(dev)[None, :]
            m = torch.rand(1, L, generator=g).to(dev) < rate
            if not bool(m.any()):
                continue
            x = torch.where(m, torch.full_like(x0, mask_id), x0)
            base = entropies(model, x)[0]
            avail = m[0].nonzero().flatten()
            for _ in range(min(args.probes, len(avail))):
                j = int(avail[torch.randint(0, len(avail), (1,), generator=g)])
                xr = x.clone()
                xr[0, j] = x0[0, j]
                drop = base - entropies(model, xr)[0]
                dist = (torch.arange(L, device=dev) - j).abs().clamp(max=D)
                keep = m[0] & (dist > 0)
                tot += torch.bincount(dist[keep], weights=drop[keep].float(), minlength=D + 1)
                cnt += torch.bincount(dist[keep], minlength=D + 1)
        mean = (tot / cnt.clamp(min=1)).cpu()
        profile = {}
        for d in range(1, D + 1):
            if float(cnt[d]) > 0:
                profile[d] = float(mean[d])
                rows.add(rate=rate, distance=d, kernel=profile[d], pairs=int(cnt[d]))
        print(f"DECAY rate={rate}: the kernel falls to a tenth of its value at distance 1 "
              f"by distance {decay_length(profile)}", flush=True)
    rows.write()


if __name__ == "__main__":
    main()
