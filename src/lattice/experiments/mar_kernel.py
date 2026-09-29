"""The kernel on MAR-B's token grid, read before any decoding.

    python -m lattice.experiments.mar_kernel --mar DATA/mar/mar --imagenet DATA/mar/imagenet_train --out DIR

For each of 64 ImageNet training images (random classes, one random image each, resized and
centre-cropped to 256 x 256) the KL-16 tokens are taken, and at each revealed fraction (0.25, 0.5,
0.75) a random subset of tokens is revealed, eight times over. One revealed token is moved by half
the image's token standard deviation in a random direction, and the relative change of the
decoder output z (what MAR's per-token diffusion head conditions on, so a moved z is a moved
conditional law) is recorded at every masked token against its Chebyshev grid distance to the
moved token. Writes OUT/kernel.csv (fraction, distance, relative change), one row per masked
token and perturbation, and prints the mean at distances 1 to 8.

    ... --information [--draws 8] [--head-samples 32] [--cfg 2.9]

reads the information kernel in place of the displacement: in each of the same states one masked
token j is chosen at random and --draws values of it are drawn from MAR's diffusion head; for every
other masked token i, --head-samples draws of it are taken from the head with j masked and with j
revealed at each drawn value, a diagonal Gaussian is fitted to each set, and the KL of the revealed
fit from the masked one, averaged over the drawn values, estimates the conditional mutual
information of i and j in bits, up to the Gaussian fit. Both sets of every token use the same
diffusion noise, so a revealed j that changed nothing would read zero. The head samples as MAR's
sampler does, guidance --cfg on its linear schedule at the state's revealed fraction. Writes
OUT/information.csv (fraction, distance, mi_bits), one row per state and token i.

    ... --floor KERNEL.csv --out DIR

writes DIR/floors.csv: every rule's first-order floor (sampling/mar_orders.py sequence_floor) at
8, 16, 32, 64 and 256 steps from a kernel's means at revealed fraction 0.5, 64 orders per rule.
"""
import argparse
import collections
import contextlib
import csv
import glob
import math
import os
import random
import statistics as st
import zlib

import numpy as np
import torch
from PIL import Image

from ..models.mar import load
from ..sampling.mar_orders import RULES, orders_for, sequence_floor

SIDE = 16


def load_image(path):
    """The image resized so its short side is 256 (bicubic) and centre-cropped, in [-1, 1]."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = 256 / min(w, h)
    im = im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BICUBIC)
    w, h = im.size
    left, top = (w - 256) // 2, (h - 256) // 2
    im = im.crop((left, top, left + 256, top + 256))
    return torch.tensor(np.asarray(im)).float().permute(2, 0, 1) / 127.5 - 1


def gaussian_kl(a, b):
    """KL(N_a || N_b) in nats of diagonal Gaussians fitted to samples a and b (..., m, dims)."""
    ma, va = a.mean(-2), a.var(-2) + 1e-8
    mb, vb = b.mean(-2), b.var(-2) + 1e-8
    return 0.5 * (va / vb + (mb - ma) ** 2 / vb - 1 + torch.log(vb / va)).sum(-1)


def head_samples(model, zc, zu, m, cfg, seed, autocast):
    """m draws per row of zc from MAR's diffusion head, guided by the unconditional zu at scale cfg
    as its sampler guides them (under autocast, as MAR samples), the diffusion noise drawn from seed
    on a forked generator so the caller's random stream is untouched: (rows, m, token dims)."""
    n = zc.shape[0]
    with torch.random.fork_rng(devices=[zc.device] if zc.is_cuda else []):
        torch.manual_seed(seed)
        with autocast:
            out = model.diffloss.sample(torch.cat([zc.repeat_interleave(m, 0), zu.repeat_interleave(m, 0)]), 1.0, cfg)
    return out[:n * m].view(n, m, -1).float()


def kernel_means(path):
    """{(fraction, distance): mean} of a kernel file, raw rows or already averaged."""
    agg = collections.defaultdict(list)
    for r in csv.DictReader(line for line in open(path) if not line.startswith("#")):
        d = int(r.get("distance") or r.get("d"))
        v = float(next(r[k] for k in ("mi_bits", "rel_displacement", "displacement") if k in r))
        agg[(round(float(r["fraction"]), 6), d)].append(v)
    return {k: st.mean(v) for k, v in agg.items()}


def floors(kernel_path, out, n_orders=64, seed=0):
    means = kernel_means(kernel_path)
    kern = {d: v for (f, d), v in means.items() if abs(f - 0.5) < 1e-9 and d <= 8}
    rows = []
    for rule in RULES:
        o = orders_for(rule, n_orders, SIDE, seed).numpy()
        for T in (8, 16, 32, 64, 256):
            rows.append(dict(rule=rule, steps=T, floor=st.mean(sequence_floor(x, kern, T) for x in o)))
            print(f"{rule:9s} {T:4d} {rows[-1]['floor']:.4f}", flush=True)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "floors.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["rule", "steps", "floor"])
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mar", required=True, help="the checkout of the authors' MAR repository")
    ap.add_argument("--imagenet", required=True, help="ImageNet-1k train/, one directory per class")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-images", type=int, default=64)
    ap.add_argument("--perturb-per-image", type=int, default=8)
    ap.add_argument("--fractions", default="0.25,0.5,0.75", help="revealed fractions of the tokens")
    ap.add_argument("--eps", type=float, default=0.5, help="the move, as a fraction of the token std")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--information", action="store_true", help="the information kernel (see above)")
    ap.add_argument("--draws", type=int, default=8, help="drawn values of the revealed token j")
    ap.add_argument("--head-samples", type=int, default=32, help="head draws per token and condition")
    ap.add_argument("--cfg", type=float, default=2.9, help="guidance scale of the head's draws")
    ap.add_argument("--floor", default="", help="a kernel file: write every rule's floor from it")
    args = ap.parse_args(argv)
    if args.floor:
        return floors(args.floor, args.out)
    dev = args.device
    # the generators are seeded before the model is built: building draws its initial weights
    # from torch's generator, and the reveals and moves below continue that stream
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model, vae = load(args.mar, dev)
    classes = sorted(os.listdir(args.imagenet))
    rng = random.Random(args.seed)
    picks = []
    while len(picks) < args.n_images:
        c = rng.randrange(1000)
        files = glob.glob(os.path.join(args.imagenet, classes[c], "*.JPEG"))
        if files:
            picks.append((c, rng.choice(files)))
    autocast = torch.autocast("cuda") if str(dev).startswith("cuda") else contextlib.nullcontext()
    rr, cc = np.divmod(np.arange(SIDE * SIDE), SIDE)
    rows = []
    with torch.no_grad():
        for c, path in picks:
            x = load_image(path).unsqueeze(0).to(dev)
            tok = vae.encode(x).sample().mul_(0.2325)                  # (1, 16, 16, 16)
            tok = tok.reshape(1, 16, SIDE * SIDE).permute(0, 2, 1)     # (1, 256, 16), one row per token
            std = tok.std()
            emb = model.class_emb(torch.tensor([c]).to(dev))
            for frac in [float(f) for f in args.fractions.split(",")]:
                n_rev = int(round(frac * SIDE * SIDE))
                for state in range(args.perturb_per_image):
                    perm = torch.randperm(SIDE * SIDE)
                    revealed, masked = perm[:n_rev], perm[n_rev:]
                    mask = torch.ones(1, SIDE * SIDE).to(dev)
                    mask[0, revealed] = 0
                    base = tok.clone()
                    base[0, masked] = 0
                    j = int(revealed[torch.randint(len(revealed), (1,))])
                    moved = base.clone()
                    moved[0, j] += args.eps * std * torch.randn(16).to(dev)
                    if args.information:
                        # after the displacement read's own draws, so the states are that read's states
                        key = zlib.crc32(f"{os.path.basename(path)}|{frac}|{state}".encode())
                        rows += information_rows(model, base, mask, masked, emb, frac, args, dev, autocast,
                                                 seed=key, rr=rr, cc=cc)
                        continue
                    with autocast:
                        z0 = model.forward_mae_decoder(model.forward_mae_encoder(base, mask, emb), mask)
                        z1 = model.forward_mae_decoder(model.forward_mae_encoder(moved, mask, emb), mask)
                    z0, z1 = z0.float()[0], z1.float()[0]
                    rel = ((z1 - z0).norm(dim=-1) / (z0.norm(dim=-1) + 1e-6)).cpu().numpy()
                    d = np.maximum(np.abs(rr - rr[j]), np.abs(cc - cc[j]))
                    for i in masked.tolist():
                        rows.append((frac, int(d[i]), float(rel[i])))
    os.makedirs(args.out, exist_ok=True)
    name, column = ("information.csv", "mi_bits") if args.information else ("kernel.csv", "rel_displacement")
    with open(os.path.join(args.out, name), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fraction", "distance", column])
        w.writerows(rows)
    agg = collections.defaultdict(list)
    for f, d, r in rows:
        agg[(f, d)].append(r)
    print("fraction | kernel at distance 1..8 (x1e-3) | divided by its value at distance 1")
    for f in sorted({k[0] for k in agg}):
        vals = [st.mean(agg[(f, d)]) if (f, d) in agg else float("nan") for d in range(1, 9)]
        ratios = " ".join(f"{v / vals[0]:.2f}" for v in vals) if vals[0] else "-"
        print(f"{f:.2f} | " + " ".join(f"{v * 1e3:6.2f}" for v in vals) + " | " + ratios)
    print(len(rows), "readings", flush=True)


def information_rows(model, base, mask, masked, emb, frac, args, dev, autocast, seed, rr, cc):
    """The information kernel's rows of one state: (fraction, distance to j, bits) for every masked i != j."""
    g = torch.Generator().manual_seed(seed)
    j = int(masked[torch.randint(len(masked), (1,), generator=g)])
    targets = torch.tensor([i for i in masked.tolist() if i != j])
    cfg = 1 + (args.cfg - 1) * frac                      # MAR's linear schedule at this revealed fraction
    null = model.fake_latent

    def z_of(tokens, m):
        with autocast:
            zc = model.forward_mae_decoder(model.forward_mae_encoder(tokens, m, emb), m)
            zu = model.forward_mae_decoder(model.forward_mae_encoder(tokens, m, null), m)
        return zc.float()[0], zu.float()[0]

    zc, zu = z_of(base, mask)
    xj = head_samples(model, zc[j:j + 1], zu[j:j + 1], args.draws, cfg, seed + 1, autocast)[0]  # (draws, dims)
    before = head_samples(model, zc[targets], zu[targets], args.head_samples, cfg, seed + 2, autocast)
    kl = torch.zeros(len(targets), device=dev)
    for k in range(args.draws):
        tokens, m = base.clone(), mask.clone()
        tokens[0, j] = xj[k].to(tokens.dtype)
        m[0, j] = 0
        zc1, zu1 = z_of(tokens, m)
        after = head_samples(model, zc1[targets], zu1[targets], args.head_samples, cfg, seed + 2, autocast)
        kl += gaussian_kl(after, before)
    bits = (kl / args.draws / math.log(2)).cpu().numpy()
    d = np.maximum(np.abs(rr - rr[j]), np.abs(cc - cc[j]))
    return [(frac, int(d[i]), float(b)) for i, b in zip(targets.tolist(), bits)]


if __name__ == "__main__":
    main()
