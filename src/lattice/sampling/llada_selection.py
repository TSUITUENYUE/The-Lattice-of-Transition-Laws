"""Selection rules for LLaDA's masked decoder: which masked positions a decoding step commits.

pick(rule, masked, k, conf, gen, min_distance) returns, for every row, k positions and a mask of those
committed. The rules, by the paper's names:

    random                     a uniformly random subset of the masked positions
    confidence                 the k masked positions of highest confidence
    spread                     farthest point first: each pick is the masked position farthest from
                               the revealed positions and from the step's earlier picks
    midpoint                     the midpoints of the k longest masked runs, longest first; with fewer
                               runs than k the step is topped up by farthest point
    confidence_min_distance  confidence with a minimum distance between the step's positions
    confidence_separator     confidence with at most one position per masked run

Both constrained rules are constraint first, confidence top-up. confidence_min_distance takes
the confidence ranking greedily, skipping positions closer than min_distance to a position already
taken in the step; confidence_separator takes each masked run's most confident position (the
leftmost on ties), runs in order of that confidence. When the constraint leaves fewer than k
positions, the step is filled to exactly k with the most confident masked positions left (held
min_distance apart while possible, then any), so every rule spends the same number of steps and the
constraint is relaxed in that step. constraint_diagnostics() records, for every step, whether
the committed positions break either constraint.

Every rule runs in tensor form on the decode's device: a rule written as a loop over rows in
Python would add its own interpreter time to the selection time the paper reports.
"""
from __future__ import annotations

import torch

NEG = -1e30
RULES = ("random", "confidence", "spread", "midpoint", "confidence_min_distance",
         "confidence_separator")
# the rules that read the model's confidence; the others choose by position alone
CONFIDENCE_RULES = ("confidence", "confidence_min_distance", "confidence_separator")


def pick(rule, masked, k, conf, gen, min_distance=0):
    """(indices, keep): k positions per row of the (B, L) mask `masked` and which are committed.

    conf is the (B, L) confidence of each position (read only by CONFIDENCE_RULES), gen the
    generator of the random rule, min_distance the minimum distance of the constrained rules.
    """
    B, L = masked.shape
    dev = masked.device
    ar = torch.arange(L, device=dev)
    if rule == "random":
        score = torch.rand(B, L, device=dev, generator=gen)
    elif rule == "confidence":
        score = conf
    elif rule == "midpoint":
        # one position per maximal masked run, the run's midpoint, longest runs first
        m = masked.long()
        start = torch.diff(m, prepend=torch.zeros(B, 1, dtype=m.dtype,
                                                  device=dev)) == 1
        rid = start.cumsum(1) * m
        n = int(rid.max().item()) + 1
        cnt = torch.zeros(B, n, device=dev)
        cnt.scatter_add_(1, rid, m.float())
        pos = torch.zeros(B, n, device=dev)
        pos.scatter_add_(1, rid, ar.float().expand(B, L) * m)
        mid = (pos / cnt.clamp(min=1)).round()
        is_mid = masked & (ar[None, :] == mid.gather(1, rid))
        score = torch.where(is_mid, cnt.gather(1, rid),
                            torch.full_like(cnt.gather(1, rid), NEG))
        kk = min(k, L)
        top = score.topk(kk, dim=1)
        idx, keep = top.indices.clone(), top.values > 0
        if not bool(keep.all()):
            # fewer runs than k: top up by farthest point, the distance to the revealed
            # positions and to the picks already made, so every step commits exactly k
            big = float(L + 1)
            idxs = ar.float()
            taken = torch.zeros(B, L, dtype=torch.bool, device=dev)
            taken.scatter_(1, idx, keep)
            sep = ~masked | taken
            posl = torch.where(sep, idxs, torch.full_like(idxs, -big))
            left = torch.cummax(posl, dim=1).values
            posr = torch.where(sep, idxs, torch.full_like(idxs, 3 * big))
            right = torch.flip(torch.cummin(torch.flip(posr, [1]),
                                            dim=1).values, [1])
            d = torch.minimum((idxs - left).abs(), (right - idxs).abs())
            d = torch.where(masked & ~taken, d, torch.full_like(d, NEG))
            for j in range(kk):
                fill = ~keep[:, j]
                if not bool(fill.any()):
                    continue
                q = d.argmax(1)
                ok = fill & (d.gather(1, q[:, None]).squeeze(1) > NEG / 2)
                idx[:, j] = torch.where(ok, q, idx[:, j])
                keep[:, j] = keep[:, j] | ok
                near = torch.minimum(
                    d, (idxs[None, :] - q[:, None].float()).abs())
                d = torch.where(ok[:, None], near, d)
                d = torch.where(ok[:, None] & (ar[None, :] == q[:, None]),
                                torch.full_like(d, NEG), d)
        return idx, keep
    elif rule == "confidence_min_distance":
        # the confidence ranking taken greedily, skipping positions within min_distance of a position
        # already taken in this step; when none is left the step is topped up with the most
        # confident masked positions remaining, so every step commits exactly k
        kk = min(k, L)
        base = torch.where(masked, conf, torch.full_like(conf, NEG))
        sc = base.clone()
        idxs = torch.arange(L, device=dev).float()
        taken = torch.zeros(B, L, dtype=torch.bool, device=dev)
        out = torch.zeros(B, kk, dtype=torch.long, device=dev)
        keep = torch.zeros(B, kk, dtype=torch.bool, device=dev)
        for j in range(kk):
            pmax = sc.argmax(1)
            spaced = sc.gather(1, pmax[:, None]).squeeze(1) > NEG / 2
            rest = torch.where(taken, torch.full_like(base, NEG), base)
            p = torch.where(spaced, pmax, rest.argmax(1))
            out[:, j] = p
            keep[:, j] = masked.gather(1, p[:, None]).squeeze(1) & \
                ~taken.gather(1, p[:, None]).squeeze(1)
            taken.scatter_(1, p[:, None], True)
            near = (idxs[None, :] - p[:, None].float()).abs() < min_distance
            sc = torch.where(near | taken, torch.full_like(sc, NEG), sc)
        return out, keep
    elif rule == "confidence_separator":
        # each masked run's most confident position (the leftmost on ties), runs taken in order
        # of that confidence; with fewer runs than k the step is topped up as the minimum-
        # distance rule tops up: the most confident masked positions left, held min_distance apart,
        # then any. On a single run it selects exactly what confidence_min_distance selects.
        kk = min(k, L)
        cm = torch.where(masked, conf, torch.full_like(conf, NEG))
        m = masked.long()
        start = torch.diff(m, prepend=torch.zeros(B, 1, dtype=m.dtype,
                                                  device=dev)) == 1
        rid = start.cumsum(1) * m
        n = int(rid.max().item()) + 1
        best = torch.full((B, n), NEG, device=dev).scatter_reduce(
            1, rid, cm, reduce="amax", include_self=True)
        cand = masked & (cm >= best.gather(1, rid))
        pos = torch.where(cand, ar.expand(B, L), torch.full_like(rid, L))
        first = torch.full((B, n), L, dtype=torch.long,
                           device=dev).scatter_reduce(
            1, rid, pos, reduce="amin", include_self=True)
        is_pick = masked & (ar[None, :] == first.gather(1, rid))
        score = torch.where(is_pick, cm, torch.full_like(cm, NEG))
        top = score.topk(kk, dim=1)
        idx, keep = top.indices.clone(), top.values > NEG / 2
        if not bool(keep.all()):
            idxs = ar.float()
            taken = torch.zeros(B, L, dtype=torch.bool, device=dev)
            taken.scatter_(1, idx, keep)
            sc = cm.clone()
            for j in range(kk):
                near = (idxs[None, :] - idx[:, j:j + 1].float()).abs() < min_distance
                sc = torch.where(near & keep[:, j:j + 1],
                                 torch.full_like(sc, NEG), sc)
            sc = torch.where(taken, torch.full_like(sc, NEG), sc)
            for j in range(kk):
                fill = ~keep[:, j]
                if not bool(fill.any()):
                    continue
                pmax = sc.argmax(1)
                spaced = sc.gather(1, pmax[:, None]).squeeze(1) > NEG / 2
                rest = torch.where(taken, torch.full_like(cm, NEG), cm)
                q = torch.where(spaced, pmax, rest.argmax(1))
                ok = fill & masked.gather(1, q[:, None]).squeeze(1) \
                    & ~taken.gather(1, q[:, None]).squeeze(1)
                idx[:, j] = torch.where(ok, q, idx[:, j])
                keep[:, j] = keep[:, j] | ok
                taken = taken | ((ar[None, :] == q[:, None]) & ok[:, None])
                near = (idxs[None, :] - q[:, None].float()).abs() < min_distance
                sc = torch.where((near & ok[:, None]) | taken,
                                 torch.full_like(sc, NEG), sc)
        return idx, keep
    elif rule == "spread":
        # farthest point first: each pick maximises its distance to the revealed positions and
        # to the picks already made in this step
        kk = min(k, L)
        big = float(L + 1)
        rev = ~masked
        idxs = torch.arange(L, device=dev).float()
        d = torch.full((B, L), big, device=dev)
        if rev.any():
            pos = torch.where(rev, idxs, torch.full_like(idxs, -big))
            left = torch.cummax(pos, dim=1).values
            pos_r = torch.where(rev, idxs, torch.full_like(idxs, 3 * big))
            right = torch.flip(torch.cummin(torch.flip(pos_r, [1]),
                                            dim=1).values, [1])
            d = torch.minimum((idxs - left).abs(), (right - idxs).abs())
        d = torch.where(masked, d, torch.full_like(d, NEG))
        out = torch.zeros(B, kk, dtype=torch.long, device=dev)
        for j in range(kk):
            p = d.argmax(1)
            out[:, j] = p
            d = torch.minimum(d, (idxs[None, :] - p[:, None].float()).abs())
            d.scatter_(1, p[:, None], NEG)
            d = torch.where(masked, d, torch.full_like(d, NEG))
        keep = masked.gather(1, out)
        return out, keep
    else:
        raise ValueError(rule)
    score = torch.where(masked, score, torch.full_like(score, NEG))
    kk = min(k, L)
    idx = score.topk(kk, dim=1).indices
    keep = masked.gather(1, idx)
    return idx, keep


def constraint_diagnostics(masked, indices, keep, min_distance):
    """For each row, whether the committed positions break either constraint.

    Runs are the maximal masked runs before the step. close_pairs counts pairs of committed
    positions closer than min_distance; extra_in_same_run counts committed positions beyond the first
    in their run.
    """
    rows = []
    for b in range(masked.shape[0]):
        positions = sorted(set(indices[b][keep[b]].detach().cpu().tolist()))
        run_of, run, inside = [], 0, False
        for value in masked[b].detach().cpu().tolist():
            if value and not inside:
                run += 1
            run_of.append(run if value else 0)
            inside = value
        runs = [run_of[p] for p in positions]
        close = sum(q - p < min_distance for i, p in enumerate(positions) for q in positions[i + 1:])
        extra = len(runs) - len(set(runs))
        rows.append(dict(selected=len(positions), close_pairs=close, extra_in_same_run=extra,
                         breaks_min_distance=bool(close), breaks_separator=bool(extra)))
    return rows
