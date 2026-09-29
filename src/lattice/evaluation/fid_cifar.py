"""FID on CIFAR-10: torch-fidelity's Inception features against the 50,000 training images.

Every CIFAR-10 number in the paper is read by this one function against the one reference
directory that data/prepare_cifar10.py writes, with the reference features cached once. Samples
reach it as 8-bit PNGs written by ``write_png``.
"""
from __future__ import annotations

from pathlib import Path

import torch


def write_png(x, out_dir, start=0):
    """Images in [-1, 1] to 8-bit PNGs named by their index from ``start``; returns the next
    index."""
    from PIL import Image
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8) \
        .permute(0, 2, 3, 1).cpu().numpy()
    for i, im in enumerate(arr):
        Image.fromarray(im).save(out_dir / f"{start + i:06d}.png")
    return start + len(arr)


def fid(sample_dir, reference_dir, cache_dir, cache_name="cifar10_train", cuda=True):
    from torch_fidelity import calculate_metrics
    return calculate_metrics(
        input1=str(sample_dir), input2=str(reference_dir), fid=True, cuda=cuda, verbose=False,
        cache_root=str(cache_dir), cache=True, input2_cache_name=cache_name,
    )["frechet_inception_distance"]
