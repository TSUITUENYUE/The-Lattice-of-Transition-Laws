"""Paths over the per-patch levels of an image, and the walk that composes a trained model along
one.

A state of the lattice is a level index per patch, 0 at the noisy end and L (the number of
transitions) at the clean end. A schedule is a path through such states:
  synchronised   every patch at one level: the L + 1 states of a diffusion sampler, L calls
  sequential     one patch at a time from the noisy end to the clean end, the rest held
  sloped         each patch lags the one before it by ``slope`` levels, so the path is
                 synchronised at slope 0 and moves towards sequential as the slope grows;
                 L + slope (P - 1) calls on P patches
A sequential or sloped path visits the patches in raster order, or in the order of a permutation;
the spread order (``spread_order``) is the bit-reversed order of the patch grid.

On the sequential path every patch after the first begins with a step to the state it already
holds (its level 0), and the walk spends a call on it that moves nothing: the path has
P (L + 1) - 1 steps, on 16 patches and 128 transitions 2063 = 2048 transitions and 15 such calls.
The paper's step counts include these calls.

Every step of the walk is one model call at the levels the path puts the patches at, followed by
the stochastic coupling of the transition, pixel by pixel: from noise variance a to b <= a, with
r = b / a, y_b = r y + (1 - r) x_hat + sqrt(b (a - b) / a) z. A patch the step holds has a = b and
keeps its value.
"""
from __future__ import annotations

import math

import torch

from ..models.pixel import level_map

SCHEDULES = ("synchronised", "sequential", "sloped")


def spread_order(n_patches):
    """Patch id visited at each position of a spread-order path: the bit-reversed Morton order
    over the patch grid. Position p of a sequential or sloped path is patch ``order[p]``."""
    side = int(math.isqrt(n_patches))
    bits = int(math.log2(side))
    out = []
    for k in range(side * side):
        m = int(format(k, f"0{2 * bits}b")[::-1], 2)
        r = c = 0
        for b in range(bits):
            r |= ((m >> (2 * b + 1)) & 1) << b
            c |= ((m >> (2 * b)) & 1) << b
        out.append(r * side + c)
    return torch.tensor(out)


def _positions(kind, n_patches, transitions, slope):
    """Per-patch level indices 0 .. transitions along the path, its starting state first, in raster
    order."""
    if kind == "synchronised":
        for k in range(transitions + 1):
            yield [k] * n_patches
        return
    if kind == "sequential":
        cur = [0] * n_patches
        for p in range(n_patches):
            # for p > 0, k = 0 repeats the previous state: the call that moves nothing
            for k in range(transitions + 1):
                cur = list(cur)
                cur[p] = k
                yield cur
        return
    lag = slope * (n_patches - 1)
    for t in range(transitions + lag + 1):
        yield [min(max(t - slope * p, 0), transitions) for p in range(n_patches)]


def path_levels(kind, n_patches, transitions, slope=1, order=None):
    """The sequence of per-patch level (knot) indices a schedule walks, its starting state first.

    ``order`` maps a position along the path to the patch id that sits there; None is raster
    order, the identity.
    """
    if kind not in SCHEDULES:
        raise ValueError(f"unknown schedule {kind}")
    for pos in _positions(kind, n_patches, transitions, slope):
        lv = torch.tensor(pos)
        if order is not None:
            cur = torch.zeros(n_patches, dtype=torch.long)
            cur[order] = lv
            lv = cur
        yield lv


def path_calls(kind, n_patches, transitions, slope=1):
    """The number of model calls a walk along the path makes (its number of steps)."""
    return sum(1 for _ in path_levels(kind, n_patches, transitions, slope)) - 1


@torch.no_grad()
def walk(model, onehot, kind, levels, patches, n_patches, slope=1,
         generator=None, device="cuda", resolution=32, order=None, record=None):
    """Compose the model along a lattice path, one call per step; returns the last estimate of
    the clean image and the number of calls.

    ``levels`` holds the noise variances of the level indices (core/levels.py). Every step reads
    the same trained model at whatever levels the schedule puts the patches at. The starting noise
    and every bridge draw come from ``generator``, a CPU generator, and move to ``device``.
    ``record(t, y, lv)``, when given, sees the state y after path index t and the per-patch level
    indices lv it sits at (at t = 0 the starting noise).
    """
    knots = torch.tensor([math.sqrt(float(k)) for k in levels.knots], device=device)
    n = onehot.shape[0]
    y = knots[0] * torch.randn((n, 3, resolution, resolution), generator=generator).to(device)
    prev, x_hat, calls = None, None, 0
    for t, lv in enumerate(path_levels(kind, n_patches, len(levels), slope, order=order)):
        lv = lv.to(device)
        if prev is None:
            prev = lv
            if record is not None:
                record(0, y, lv)
            continue
        smap = level_map(prev.unsqueeze(0).expand(n, -1), patches, knots)
        x_hat = model.denoise(y, smap, onehot)
        calls += 1
        a = smap ** 2
        b = level_map(lv.unsqueeze(0).expand(n, -1), patches, knots) ** 2
        r = (b / a).clamp(max=1.0)
        tau = (b * (a - b) / a).clamp_min(0)
        y = r * y + (1 - r) * x_hat + tau.sqrt() * torch.randn(y.shape, generator=generator).to(device)
        prev = lv
        if record is not None:
            record(t, y, lv)
    return x_hat if x_hat is not None else y, calls
