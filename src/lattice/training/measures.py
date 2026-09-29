"""Training distributions over the lattice points of an image, drawn as per-patch level indices
(0 at the noisy end, L = ``transitions`` at the clean end).

  iid            every patch's level drawn independently and uniformly from 0..L, so the model
                 sees the whole lattice and reaches a corner with vanishing probability
  mixed          per batch, a quarter at one common level (the synchronised corner), a quarter
                 on the patch-sequential path, and half at independent levels
  synchronised   one level for the whole image, which is a diffusion model's training
  sequential     states of the patch-sequential path: patches before a front are clean, the
                 patch at the front is at a uniform level, the patches after it are at the noisy
                 end, the front uniform over the patches

The CIFAR-10 and ImageNet-256 trainers draw from the same function.
"""
from __future__ import annotations

import torch

MEASURES = ("iid", "mixed", "synchronised", "sequential")


def level_drawer(measure, transitions, n_patches, gen, dev):
    """``draw(n)``: n states of ``measure`` as (n, n_patches) level indices on ``dev``, drawn from
    the CPU generator ``gen``."""
    if measure not in MEASURES:
        raise ValueError(f"unknown measure {measure}")

    def draw(n):
        if measure == "synchronised":
            k = torch.randint(0, transitions + 1, (n, 1), generator=gen)
            return k.expand(-1, n_patches).to(dev)
        if measure == "mixed":
            k = torch.randint(0, transitions + 1, (n, n_patches), generator=gen)
            q = n // 4
            ks = torch.randint(0, transitions + 1, (q, 1), generator=gen)
            k[:q] = ks.expand(-1, n_patches)
            front = torch.randint(0, n_patches, (q, 1), generator=gen)
            kf = torch.randint(0, transitions + 1, (q, 1), generator=gen)
            p = torch.arange(n_patches).view(1, -1)
            k[q:2 * q] = torch.where(
                p < front, torch.full((q, n_patches), transitions, dtype=k.dtype),
                torch.where(p == front, kf.expand(-1, n_patches),
                            torch.zeros(q, n_patches, dtype=k.dtype)))
            return k.to(dev)
        if measure == "sequential":
            front = torch.randint(0, n_patches, (n, 1), generator=gen)
            k = torch.randint(0, transitions + 1, (n, 1), generator=gen)
            p = torch.arange(n_patches).view(1, -1)
            return torch.where(p < front, transitions, torch.where(p == front, k, 0)).to(dev)
        return torch.randint(0, transitions + 1, (n, n_patches), generator=gen).to(dev)
    return draw
