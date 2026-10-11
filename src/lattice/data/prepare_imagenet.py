"""Encode the ImageNet-1k training set into the two latent banks the SiT networks train on.

    python -m lattice.data.prepare_imagenet --train-dir IMAGENET/train --out DATA/imagenet

Every image gets ADM's centre crop to 256 x 256 (the crop of guided-diffusion, DiT and SiT) and is
encoded by sd-vae-ft-ema, together with its mirror image from the same crop, so the two banks are
index-aligned by construction. Files are taken in sorted order, the class being the index of the
sorted synset directory. Writes OUT/latents/shard_*.pt and OUT/latents_flip/shard_*.pt (10,000
images per shard, resumable), then packs each bank into OUT/latents.pt and OUT/latents_flip.pt
and checks the packed files against the shards.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .imagenet_latents import load_vae


# center_crop_arr is taken from openai/guided-diffusion (guided_diffusion/image_datasets.py).
# Copyright (c) 2021 OpenAI
# Licensed under the MIT License; the licence text is in LICENSES/guided-diffusion-MIT.txt at the
# root of this repository.
def center_crop_arr(pil_image, image_size=256):
    """ADM's crop: box-downsample by two while the short side is at least twice the size,
    bicubic-resize the short side to the size, then take the centre square."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


class _Crops(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, _ = self.items[i]
        a = center_crop_arr(Image.open(f).convert("RGB"), 256)
        return torch.from_numpy(np.asarray(a, dtype=np.uint8).copy())


def _save(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def encode(train_dir, out, shard=10000, bs=100, processes=8, shards=""):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    vae = load_vae(dev)
    root = Path(train_dir)
    wnids = sorted(d.name for d in root.iterdir() if d.is_dir())
    cls_of = {w: i for i, w in enumerate(wnids)}
    files = [(f, cls_of[w]) for w in wnids for f in sorted((root / w).iterdir())]
    a, b = Path(out) / "latents", Path(out) / "latents_flip"
    a.mkdir(parents=True, exist_ok=True)
    b.mkdir(parents=True, exist_ok=True)
    listing = "\n".join(f"{f.relative_to(root)}\t{c}" for f, c in files)
    for d in (a, b):
        (d / "files.md5").write_text(f"{hashlib.md5(listing.encode()).hexdigest()}  {len(files)} files, crop adm\n")
    n_shards = (len(files) + shard - 1) // shard
    print(f"encode: {len(files)} files, {len(wnids)} classes, {n_shards} shards", flush=True)
    lo, hi = (int(x) for x in shards.split(":")) if shards else (0, n_shards)
    t0 = time.time()
    for s in range(lo, min(hi, n_shards)):
        fa, fb = a / f"shard_{s:05d}.pt", b / f"shard_{s:05d}.pt"
        if fa.exists() and fb.exists():
            continue
        chunk = files[s * shard:(s + 1) * shard]
        mus, lvs, mus2, lvs2 = [], [], [], []
        loader = torch.utils.data.DataLoader(_Crops(chunk), batch_size=bs, num_workers=processes, shuffle=False)
        for x in loader:
            x = x.permute(0, 3, 1, 2).float().to(dev) / 127.5 - 1.0
            d = vae.encode(x).latent_dist
            mus.append(d.mean.half().cpu())
            lvs.append(d.logvar.half().cpu())
            d2 = vae.encode(x.flip(3)).latent_dist
            mus2.append(d2.mean.half().cpu())
            lvs2.append(d2.logvar.half().cpu())
        c = torch.tensor([c for _, c in chunk], dtype=torch.int16)
        _save({"mu": torch.cat(mus), "logvar": torch.cat(lvs), "c": c}, fa)
        _save({"mu": torch.cat(mus2), "logvar": torch.cat(lvs2), "c": c}, fb)
        print(f"shard {s + 1}/{n_shards} ({time.time() - t0:.0f}s)", flush=True)
    return n_shards


def pack(shard_dir, out):
    """The shards written end to end into one file, then checked row by row."""
    shards = sorted(Path(shard_dir).glob("shard_*.pt"))
    d0 = torch.load(shards[0], map_location="cpu", weights_only=False)
    keys = [k for k in ("mu", "logvar", "c") if k in d0]
    counts = [len(torch.load(s, map_location="cpu", weights_only=False, mmap=True)[keys[0]]) for s in shards]
    n = sum(counts)
    packed = {k: torch.empty((n,) + tuple(d0[k].shape[1:]), dtype=d0[k].dtype) for k in keys}
    at = 0
    for s, c in zip(shards, counts):
        d = torch.load(s, map_location="cpu", weights_only=False)
        for k in keys:
            packed[k][at:at + c] = d[k]
        at += c
    _save(packed, out)
    m = torch.load(out, map_location="cpu", weights_only=False, mmap=True)
    at = 0
    for s, c in zip(shards, counts):
        d = torch.load(s, map_location="cpu", weights_only=False)
        for j in (0, c // 2, c - 1):
            for k in keys:
                assert torch.equal(d[k][j], m[k][at + j]), (s, k, j)
        at += c
    print(f"packed {len(shards)} shards, {n} latents -> {out}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", required=True, help="ImageNet-1k train/, one directory per synset")
    ap.add_argument("--out", required=True, help="where the two banks go")
    ap.add_argument("--bs", type=int, default=100, help="images a VAE encode")
    ap.add_argument("--processes", type=int, default=8, help="processes that decode and crop the images")
    ap.add_argument("--shards", default="", help="A:B encodes shards A..B-1 only (several jobs may share a bank)")
    ap.add_argument("--skip-pack", action="store_true", help="leave the shards unpacked")
    args = ap.parse_args(argv)
    n_shards = encode(args.train_dir, args.out, bs=args.bs, processes=args.processes, shards=args.shards)
    for name in ("latents", "latents_flip"):
        have = len(list((Path(args.out) / name).glob("shard_*.pt")))
        if have < n_shards:
            print(f"{name}: {have} of {n_shards} shards so far; packing waits for all of them", flush=True)
        elif not args.skip_pack and not (Path(args.out) / f"{name}.pt").exists():
            pack(Path(args.out) / name, Path(args.out) / f"{name}.pt")


if __name__ == "__main__":
    main()
