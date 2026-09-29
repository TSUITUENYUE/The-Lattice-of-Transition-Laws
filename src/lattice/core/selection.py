"""Selection rules: which coordinates a step advances.

A decode of a sequence advances, at every step, a set of masked coordinates, and the rule that
picks the set decides what the step costs (the total correlation of the set). The rules here are
the ones the paper compares on text8 (the dilated rule, a fixed bit-reversal order, is
evaluation/text8_bpc.dilated_order):

  contiguous   the first k masked positions, in order
  random       a uniformly random subset of k masked positions
  spread       k equally spaced masked positions: anchors evenly over the masked span, each
               taking the nearest masked position not yet taken
  separator    at most one coordinate per masked run, each as far as possible from the step's
               other picks: spread over segments (below) with one segment per masked run. Distance
               is measured to the step's own picks, so inside a run it takes the end next to a
               revealed coordinate; when the runs are fewer than k it takes a second coordinate of
               a run
  midpoint     the midpoint of each masked run (Theorem 1's schedule), below
  midpoint, ties spread
               the midpoint rule with the runs tied at the shortest length it takes chosen
               spread evenly by position among the tied runs, below
  confidence   the k masked positions of highest model confidence; with a minimum distance, the
               most confident positions at least that far from every pick so far, topped up by the
               most confident remaining positions when too few qualify
  spread over segments   spread that takes at most one position per segment: the first pick is
               the middle masked position, each later pick the masked position farthest from the
               step's picks so far whose segment is untaken, and when the segments run out the
               farthest remaining position. With one segment per masked run it is the separator
               rule; with segments cut at spaces it is the word-aware spread rule of free
               generation

Each rule has a row form, which takes one row's mask (True = still masked) and returns sorted
positions, and most have a batch form, which takes a (rows, length) mask with at least k masked
positions in every row and returns (rows, k) sorted positions on the mask's device. The batch
forms pick the row forms' sets (random: the same law with other draws) and exist to be timed: a
row form loops over the rows in Python and reads positions back to the host, which a timing would
charge to the rule.

Exact ties. spread's row form orders positions equally near an anchor with the sort of the device
it runs on, and a GPU sort orders such ties differently from a CPU sort; the batch form gives
them to the lower position. confidence breaks exact ties in the confidence toward the lower
position in its batch form; its row form orders them as the device's sort does.
"""
from __future__ import annotations

import torch


# ---- row forms -------------------------------------------------------------------------------
def contiguous(mask_row, k):
    return mask_row.nonzero(as_tuple=False).view(-1)[:k]


def random(mask_row, k, generator=None):
    pos = mask_row.nonzero(as_tuple=False).view(-1)
    if len(pos) <= k:
        return pos
    perm = torch.randperm(len(pos), generator=generator)[:k]
    return pos[perm].sort().values


def spread(mask_row, k):
    pos = mask_row.nonzero(as_tuple=False).view(-1)
    if len(pos) <= k:
        return pos
    # the anchors lie on the masked span, so a revealed prefix in front of the decoded tail does
    # not pull them to one edge
    lo, hi = float(pos.min()), float(pos.max())
    taken, out = set(), []
    for t in torch.linspace(lo, hi, k).tolist():
        for j in torch.argsort((pos.float() - t).abs()).tolist():
            p = int(pos[j])
            if p not in taken:
                taken.add(p)
                out.append(p)
                break
    return torch.tensor(sorted(out), dtype=torch.long)


def confidence(mask_row, conf_row, k, min_distance=0):
    pos = mask_row.nonzero(as_tuple=False).view(-1)
    if len(pos) <= k:
        return pos
    order = pos[torch.argsort(conf_row[pos], descending=True)]
    if min_distance <= 1:
        return order[:k].sort().values
    out = []
    for p in order.tolist():
        if all(abs(p - q) >= min_distance for q in out):
            out.append(p)
            if len(out) == k:
                break
    for p in order.tolist():
        if len(out) == k:
            break
        if p not in out:
            out.append(p)
    return torch.tensor(sorted(out), dtype=torch.long)


def spread_segments(mask_row, seg_row, k):
    """Spread over segments (above) for one row; ``seg_row`` gives each position's segment."""
    pos = torch.nonzero(mask_row, as_tuple=False).flatten().tolist()
    if not pos:
        return torch.zeros(0, dtype=torch.long)
    chosen = [pos[len(pos) // 2]]
    used = {int(seg_row[chosen[0]])}
    while len(chosen) < min(k, len(pos)):
        best, best_d, best_relaxed = None, -1, None
        for p in pos:
            if p in chosen:
                continue
            d = min(abs(p - c) for c in chosen)
            if int(seg_row[p]) not in used:
                if d > best_d:
                    best, best_d = p, d
            elif best is None and (best_relaxed is None or d > best_relaxed[1]):
                best_relaxed = (p, d)
        if best is None:
            if best_relaxed is None:
                break
            best = best_relaxed[0]
        chosen.append(best)
        used.add(int(seg_row[best]))
    return torch.tensor(sorted(chosen), dtype=torch.long)


def separator(mask_row, k):
    return spread_segments(mask_row, (~mask_row).long().cumsum(0), k)


# ---- the midpoint rule -----------------------------------------------------------------------
def midpoint_indices(mask, k, first_step=False, ties="left"):
    """The midpoint rule on one row, as a list: ``mask`` is a sequence of booleans (True = masked).

    The first step of a decode advances k coordinates of the single masked run: the lower
    midpoints of k equal intervals, which needs the run's length divisible by k. Every later step
    advances the lower midpoint of each of the k longest masked runs, equal lengths taken left to
    right. A step never takes two coordinates of one run, so there is no top-up: fewer than k runs
    is an error. With ties="spread" (midpoint, ties spread) the runs longer than the k-th longest
    are all taken, and the m picks left go to the t runs of the k-th longest length, in their
    left-to-right order, at indices floor((2j + 1) t / 2m) for j = 0 .. m - 1: the centres of m
    equal intervals of the tied runs, which is how the first step places its picks in one run.
    """
    free = [i for i, value in enumerate(mask) if value]
    if k < 1 or len(free) < k:
        raise ValueError("midpoint needs 1 <= k <= the number of masked positions")
    if first_step:
        lo, n = free[0], len(free)
        if free != list(range(lo, lo + n)) or n % k:
            raise ValueError("the first midpoint step needs one masked run divisible into k intervals")
        return [lo + ((2 * j + 1) * n - k) // (2 * k) for j in range(k)]
    runs = []
    for p in free:
        if not runs or p != runs[-1][1] + 1:
            runs.append([p, p])
        else:
            runs[-1][1] = p
    if len(runs) < k:
        raise ValueError(f"midpoint has {len(runs)} masked runs for {k} picks")
    selected = sorted(runs, key=lambda run: (-(run[1] - run[0] + 1), run[0]))[:k]
    if ties == "spread":
        cut = selected[-1][1] - selected[-1][0] + 1
        longer = [run for run in runs if run[1] - run[0] + 1 > cut]
        tied = [run for run in runs if run[1] - run[0] + 1 == cut]
        m = k - len(longer)
        selected = longer + [tied[(2 * j + 1) * len(tied) // (2 * m)] for j in range(m)]
    elif ties != "left":
        raise ValueError(f"ties must be left or spread, not {ties!r}")
    return sorted((lo + hi) // 2 for lo, hi in selected)


def midpoint(mask_row, k, first_step=False, ties="left"):
    return torch.tensor(midpoint_indices(mask_row.tolist(), k, first_step, ties), dtype=torch.long)


def midpoint_schedule(length, steps, prefix=0, ties="left"):
    """Every step's set of the midpoint decode of positions prefix .. length - 1 in ``steps`` steps
    of equal width (the decoded length must be divisible by the steps); ``ties`` as in
    midpoint_indices."""
    n = length - prefix
    if steps < 1 or n % steps:
        raise ValueError("the midpoint decode needs the decoded length divisible by the steps")
    mask = [i >= prefix for i in range(length)]
    sets = []
    for step in range(steps):
        selected = midpoint_indices(mask, n // steps, first_step=(step == 0), ties=ties)
        sets.append(selected)
        for p in selected:
            mask[p] = False
    if any(mask):
        raise AssertionError("the midpoint decode left positions masked")
    return sets


# ---- batch forms -----------------------------------------------------------------------------
def contiguous_batch(mask, k):
    ar = torch.arange(mask.shape[1], device=mask.device)
    return ar.expand_as(mask).masked_fill(~mask, mask.shape[1]).topk(k, dim=1, largest=False).values


def random_batch(mask, k, generator=None):
    u = torch.rand(mask.shape, device=mask.device, generator=generator)
    return u.masked_fill(~mask, -1.0).topk(k, dim=1).indices.sort(1).values


def spread_batch(mask, k):
    B, L = mask.shape
    ar = torch.arange(L, device=mask.device, dtype=torch.float64)
    lo = mask.int().argmax(1).double()
    hi = (L - 1 - mask.flip(1).int().argmax(1)).double()
    frac = torch.arange(k, device=mask.device, dtype=torch.float64) / max(k - 1, 1)
    anchors = lo[:, None] + (hi - lo)[:, None] * frac[None, :]
    free = mask.clone()
    out = torch.empty(B, k, dtype=torch.long, device=mask.device)
    for j in range(k):
        d = (ar[None, :] - anchors[:, j:j + 1]).abs().masked_fill(~free, float("inf"))
        p = d.argmin(1)
        out[:, j] = p
        free.scatter_(1, p[:, None], False)
    return out.sort(1).values


def confidence_batch(mask, conf, k, min_distance=0):
    """With a minimum distance the most confident position spaced from every pick so far is
    taken, which is the position the row form's walk down its ranking accepts next; when none is
    left the most confident position not taken tops the step up."""
    B, L = mask.shape
    base = conf.float().masked_fill(~mask, float("-inf"))
    if min_distance <= 1:
        order = base.sort(dim=1, descending=True, stable=True).indices
        return order[:, :k].sort(1).values
    ar = torch.arange(L, device=mask.device)
    sc = base.clone()
    taken = torch.zeros_like(mask)
    out = torch.empty(B, k, dtype=torch.long, device=mask.device)
    for j in range(k):
        pmax = sc.argmax(1)
        spaced = sc.gather(1, pmax[:, None]).squeeze(1) > float("-inf")
        p = torch.where(spaced, pmax, base.masked_fill(taken, float("-inf")).argmax(1))
        out[:, j] = p
        taken.scatter_(1, p[:, None], True)
        near = (ar[None, :] - p[:, None]).abs() < min_distance
        sc = sc.masked_fill(near | taken, float("-inf"))
    return out.sort(1).values


def spread_segments_batch(mask, seg, k):
    """spread_segments for every row at once (ties to the lower position)."""
    B, L = mask.shape
    ar = torch.arange(L, device=mask.device)
    rank = mask.long().cumsum(1)
    mid = (mask & (rank == (mask.sum(1) // 2 + 1)[:, None])).int().argmax(1)
    out = torch.empty(B, k, dtype=torch.long, device=mask.device)
    out[:, 0] = mid
    free = mask.clone()
    free.scatter_(1, mid[:, None], False)
    d = (ar[None, :] - mid[:, None]).abs()
    used = seg == seg.gather(1, mid[:, None])
    for j in range(1, k):
        fresh = d.masked_fill(~(free & ~used), -1)
        rest = d.masked_fill(~free, -1)
        p = torch.where(fresh.amax(1) >= 0, fresh.argmax(1), rest.argmax(1))
        out[:, j] = p
        free.scatter_(1, p[:, None], False)
        d = torch.minimum(d, (ar[None, :] - p[:, None]).abs())
        used = used | (seg == seg.gather(1, p[:, None]))
    return out.sort(1).values


def separator_batch(mask, k):
    return spread_segments_batch(mask, (~mask).long().cumsum(1), k)


def midpoint_batch(mask, k, first_step=False):
    """The midpoint rule for every row at once, in tensor operations on the mask's device."""
    B, L = mask.shape
    dev = mask.device
    ar = torch.arange(L, device=dev)
    n = mask.sum(1)
    if bool((n < k).any()) or k < 1:
        raise ValueError("midpoint needs 1 <= k <= the number of masked positions")
    lo = mask.int().argmax(1)
    if first_step:
        hi = L - 1 - mask.flip(1).int().argmax(1)
        if bool(((hi - lo + 1) != n).any()) or bool((n % k != 0).any()):
            raise ValueError("the first midpoint step needs one masked run divisible into k intervals")
        j = torch.arange(k, device=dev)
        return lo[:, None] + ((2 * j[None, :] + 1) * n[:, None] - k) // (2 * k)
    before = torch.zeros_like(mask)
    before[:, 1:] = mask[:, :-1]
    after = torch.zeros_like(mask)
    after[:, :-1] = mask[:, 1:]
    starts = (mask & ~before).long()
    runs = starts.sum(1)
    if bool((runs < k).any()):
        raise ValueError(f"midpoint has {int(runs.min())} masked runs for {k} picks")
    r = int(runs.max())
    first = ar.expand(B, L).masked_fill(~(mask & ~before), L).sort(1).values[:, :r]
    last = ar.expand(B, L).masked_fill(~(mask & ~after), L).sort(1).values[:, :r]
    valid = torch.arange(r, device=dev)[None, :] < runs[:, None]
    # longer runs first, equal lengths from the left
    score = ((last - first + 1) * (L + 1) + (L - first)).masked_fill(~valid, -1)
    pick = score.topk(k, dim=1).indices
    return ((first.gather(1, pick) + last.gather(1, pick)) // 2).sort(1).values
