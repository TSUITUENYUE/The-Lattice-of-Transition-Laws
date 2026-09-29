"""FID-50K of the released MAR-B model under one reveal order at one number of steps.

    python -m lattice.experiments.mar_decode --mar DATA/mar/mar --rule spread --num-iter 8 --out DIR

One weight set with MAR's own chunk sizes (its cosine schedule), guidance (2.9, linear schedule),
temperature (1.0) and 100 diffusion steps per token; only the order in which the 16 x 16 tokens
are revealed changes (sampling/mar_orders.py). Images are generated the way MAR's engine_mar.py
evaluates: 50 of each of the 1,000 classes in class order, batches of 128, seed 0, clamped and
rounded to 8-bit PNGs. FID and Inception Score come from torch-fidelity as MAR's environment
installs it, against MAR's FID statistics of the ADM ImageNet-256 reference batch. Appends one row
(rule, num_iter, cfg, seed, n, fid, isc) to OUT/fid.csv, then deletes the images unless
--keep-images.

After every batch the cell saves its images and every random state it draws from (torch's CPU and
GPU generators, numpy's, and the number of orders drawn), so an interrupted cell resumes with
exactly the draws an uninterrupted one makes.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import json
import os
import shutil
import time

import numpy as np
import torch

from ..models.mar import FID_STATS, load
from ..sampling.mar_orders import RULES, install_rule


def capture_random_state(counter, device):
    state = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "rule_calls": counter["n"]}
    if str(device).startswith("cuda"):
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state, counter, device):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    counter["n"] = state["rule_calls"]
    if str(device).startswith("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mar", required=True, help="the checkout of the authors' MAR repository")
    ap.add_argument("--rule", required=True, choices=RULES)
    ap.add_argument("--num-iter", type=int, required=True, help="decoding steps")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-images", type=int, default=50000, help="a multiple of 1,000 (equal classes)")
    ap.add_argument("--bsz", type=int, default=128)
    ap.add_argument("--cfg", type=float, default=2.9, help="guidance scale")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--num-sampling-steps", default="100", help="diffusion steps per token")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--keep-images", action="store_true", help="keep the PNGs after the FID read")
    args = ap.parse_args(argv)
    if args.num_images <= 0 or args.num_images % 1000:
        ap.error("--num-images must be a positive multiple of 1,000")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tag = f"{args.rule}_iter{args.num_iter}_cfg{args.cfg}_s{args.seed}"
    folder = os.path.join(args.out, tag)
    progress = os.path.join(args.out, f".{tag}.progress.pt")
    os.makedirs(folder, exist_ok=True)
    key = dict(rule=args.rule, num_iter=args.num_iter, cfg=args.cfg, seed=args.seed, temperature=args.temperature,
               bsz=args.bsz, num_images=args.num_images, num_sampling_steps=args.num_sampling_steps)
    st = torch.load(progress, weights_only=False) if os.path.exists(progress) else None
    if st is not None and st["key"] != key:
        raise SystemExit(f"{progress} belongs to other settings; use a new output directory")
    if st is None and any(n.endswith(".png") for n in os.listdir(folder)):
        raise SystemExit(f"{folder} holds images but no saved random state; use a new output directory")
    start0 = st["next"] if st is not None else 0
    if start0 < args.num_images:
        from PIL import Image
        model, vae = load(args.mar, args.device, args.num_sampling_steps)
        counter = install_rule(model, args.rule, args.seed * 100003, args.device)
        if st is not None:
            restore_random_state(st, counter, args.device)
            print(f"{tag}: resumed at image {start0}", flush=True)
        autocast = torch.autocast("cuda") if str(args.device).startswith("cuda") else contextlib.nullcontext()
        labels = np.arange(1000).repeat(args.num_images // 1000)
        t0 = time.time()
        with torch.no_grad():
            for start in range(start0, args.num_images, args.bsz):
                lab = torch.tensor(labels[start:start + args.bsz]).long().to(args.device)
                with autocast:
                    tok = model.sample_tokens(bsz=len(lab), num_iter=args.num_iter, cfg=args.cfg,
                                              cfg_schedule="linear", labels=lab, temperature=args.temperature)
                    # 32 images at a time, so the decoder's activations for the whole batch never sit on
                    # the card together; every decoder layer acts on each image separately, so the
                    # images are the ones a whole-batch decode gives
                    img = torch.cat([vae.decode(t / 0.2325) for t in tok.split(32)])
                img = ((img + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
                for j, im in enumerate(img):
                    path = os.path.join(folder, f"{start + j:06d}.png")
                    Image.fromarray((im * 255).round().astype(np.uint8)).save(path + ".tmp", format="PNG")
                    os.replace(path + ".tmp", path)
                torch.save({"key": key, "next": start + len(lab), **capture_random_state(counter, args.device)},
                           progress + ".tmp")
                os.replace(progress + ".tmp", progress)
                if (start // args.bsz) % 20 == 0:
                    print(f"{tag}: {start + len(lab)}/{args.num_images} ({time.time() - t0:.0f}s)", flush=True)
    missing = [i for i in range(args.num_images) if not os.path.exists(os.path.join(folder, f"{i:06d}.png"))]
    if missing:
        raise SystemExit(f"{folder} lacks {len(missing)} images; cannot score an incomplete cell")
    import torch_fidelity
    metrics = torch_fidelity.calculate_metrics(
        input1=folder, input2=None, fid_statistics_file=os.path.join(os.path.abspath(args.mar), FID_STATS),
        cuda=str(args.device).startswith("cuda"), isc=True, fid=True, kid=False, prc=False, verbose=False)
    row = dict(rule=args.rule, num_iter=args.num_iter, cfg=args.cfg, seed=args.seed, n=args.num_images,
               fid=metrics["frechet_inception_distance"], isc=metrics["inception_score_mean"])
    print("RESULT", json.dumps(row), flush=True)
    with open(os.path.join(args.out, "fid.csv"), "a", newline="") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)           # cells finishing together append one at a time
        fh.seek(0, os.SEEK_END)
        w = csv.DictWriter(fh, fieldnames=list(row))
        if fh.tell() == 0:
            w.writeheader()
        w.writerow(row)
    if not args.keep_images:
        shutil.rmtree(folder)
        os.remove(progress)


if __name__ == "__main__":
    main()
