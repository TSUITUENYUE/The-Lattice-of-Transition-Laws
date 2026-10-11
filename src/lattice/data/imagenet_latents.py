"""ImageNet-256 as sd-vae-ft-ema latents: the bank the SiT networks train on.

A bank holds, for every training image, the VAE posterior's mean and log-variance (float16) and
the class index. A second, index-aligned bank holds the latents of the mirrored images, so a
horizontal flip is a flip of the image, never of the latent grid. data/prepare_imagenet.py
writes both and packs each into one file, which is read with mmap so several trainings on one
host share a single copy of its pages.
"""
from __future__ import annotations

from pathlib import Path

import torch

SCALE = 0.18215          # the latent scale every SiT/DiT pipeline applies


def read_bank(path, labels=True):
    """(mu, logvar, class) of a packed bank file, or of a directory of shard_*.pt files."""
    p = Path(path)
    if p.is_file():
        d = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        return d["mu"], d["logvar"], d["c"].long() if labels else None
    shards = sorted(p.glob("shard_*.pt"))
    mus, lvs, cls = [], [], []
    for s in shards:
        d = torch.load(s, map_location="cpu", weights_only=False)
        mus.append(d["mu"])
        lvs.append(d["logvar"])
        if labels:
            cls.append(d["c"].long())
    return (torch.cat(mus), torch.cat(lvs),
            torch.cat(cls) if labels else None)


def load_vae(device):
    """sd-vae-ft-ema, the VAE of the banks and of every read."""
    import safetensors.torch  # noqa: F401  (diffusers expects the submodule to be loaded)
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()
    return vae
