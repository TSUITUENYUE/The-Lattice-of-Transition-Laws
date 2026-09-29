"""Readers of a trained text8 network: at one lattice state, over a set, along a decoding path.

Three readings recur. The product law charges a set from one call and pays the dependence inside
it. The exact chain reveals the set one coordinate at a time and pays nothing. A decoding path
charges each step its product law, so its bits per character less the chain over the same sets is
the dependence the schedule drops. The masked-diffusion bound (the masked ELBO) is read on fixed
windows with fixed draws. Selection rules come from lattice.core.selection.

Every decode here is teacher forced: the true characters of a step's set are written into the
state before the next step, so a path's reading depends on the schedule and the model, never on
the model's own samples. Cross-entropies are summed in nats and divided by the scored count and
ln 2: bits per character. On a GPU the network runs under bfloat16 autocast; on a CPU in float32.
"""
from __future__ import annotations

import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from ..core import selection as SEL
from ..data.text8 import LN2, MASK, V, fixed_rates, mask_batch

# the rules path_bpc runs (lattice.core.selection states each); min_distance is confidence held to
# a minimum distance between the picks of a step
RULES = ("contiguous", "random", "spread", "separator", "midpoint", "midpoint_ties_spread", "dilated", "confidence",
         "min_distance")
MODEL_RULES = ("confidence", "min_distance")        # the rules that read the model's confidence


def forward_bf16(model, x, *args):
    with (torch.autocast("cuda", dtype=torch.bfloat16) if x.is_cuda else nullcontext()):
        return model(x, *args)


def logits_at(model, x, S):
    """Logits predicting the coordinates in S, whatever the network's convention.

    A network whose logits sit one position ahead of what they predict (the causal network) is
    read one position back, so a causal and a bidirectional network are read at the same state.
    """
    lg = forward_bf16(model, x)
    idx = (S - getattr(model, "SHIFT", 0)).clamp_min(0)
    return lg.gather(1, idx.unsqueeze(-1).expand(-1, -1, V))


def select_sets(mask, k, rule="random", generator=None, conf=None, min_distance=0, on_device=False):
    """The k positions a step advances in every row. ``mask`` marks what is still masked.

    Row forms select one row at a time (rows with fewer than k masked positions return what they
    have); with on_device the batch forms select all rows together, which needs k masked positions
    in every row and, for random, a generator on the mask's device.
    """
    if on_device:
        if rule == "contiguous":
            return SEL.contiguous_batch(mask, k)
        if rule == "random":
            return SEL.random_batch(mask, k, generator=generator)
        if rule == "spread":
            return SEL.spread_batch(mask, k)
        if rule == "confidence":
            return SEL.confidence_batch(mask, conf, k)
        if rule == "min_distance":
            return SEL.confidence_batch(mask, conf, k, min_distance=min_distance)
        raise ValueError(f"no batch form of the selection rule {rule}")
    out = []
    for i in range(mask.shape[0]):
        row = mask[i]
        if rule == "contiguous":
            s = SEL.contiguous(row, k)
        elif rule == "random":
            s = SEL.random(row, k, generator=generator)
        elif rule == "spread":
            s = SEL.spread(row, k)
        elif rule == "confidence":
            s = SEL.confidence(row, conf[i], k)
        elif rule == "min_distance":
            s = SEL.confidence(row, conf[i], k, min_distance=min_distance)
        else:
            raise ValueError(f"unknown selection rule {rule}")
        out.append(s)
    return torch.stack(out).to(mask.device)


def top_up(xin, mask, k, generator=None):
    """Mask extra positions so every row has k to advance, rather than dropping short rows, which
    would make the batch depend on the mask rate."""
    for i in range(mask.shape[0]):
        have = int(mask[i].sum())
        if have >= k:
            continue
        rest = (~mask[i]).nonzero(as_tuple=False).view(-1)
        add = rest[torch.randperm(len(rest), generator=generator)[:k - have]]
        mask[i, add] = True
        xin[i, add] = MASK
    return xin, mask


@torch.no_grad()
def product_bpc(net, xin, S, tokens):
    """One call, the coordinates of S charged independently: what a parallel step decodes."""
    pv = logits_at(net, xin, S)
    return float(F.cross_entropy(pv.reshape(-1, V).float(), tokens.reshape(-1))) / LN2


@torch.no_grad()
def chain_bpc(net, xin, S, tokens):
    """One call per coordinate, in the order S gives: the exact chain over S."""
    ces, cur = [], xin.clone()
    shift = getattr(net, "SHIFT", 0)
    for i in range(S.shape[1]):
        lg = forward_bf16(net, cur)
        p = S[:, i]
        row = lg.gather(1, (p - shift).clamp_min(0).view(-1, 1, 1).expand(-1, 1, V)).squeeze(1)
        ces.append(F.cross_entropy(row.float(), tokens[:, i], reduction="none"))
        cur = cur.scatter(1, p.view(-1, 1), tokens[:, i:i + 1])
    return float(torch.stack(ces).mean()) / LN2


def _sync(x):
    if x.is_cuda:
        torch.cuda.synchronize()


def dilated_order(length):
    """The dilated rule's order, fixed before decoding: the positions of the window in bit-reversal
    order of their index."""
    w = max(1, (length - 1).bit_length())
    return sorted(range(length), key=lambda i: int(format(i, f"0{w}b")[::-1], 2))


@torch.no_grad()
def path_bpc(net, x, steps=None, rule="random", generator=None, min_distance=0, prefix=0,
             widths=None, stats=None, timing=None, sets=None, on_device=False, chain_within=False):
    """Bits per character of the coordinates after the prefix under a decoding path.

    The first ``prefix`` characters of every window are revealed. With ``steps`` every step
    advances the same number of coordinates, (length - prefix) // steps, and the last step what is
    left; with ``widths`` step t advances widths[t] (a width profile summing to the decoded
    length). The named rule chooses the set; each step is charged its product law and its true
    characters are written back. N steps over N coordinates is the exact chain, one step a single
    parallel shot. The confidence rules rank the masked positions by the maximum predicted
    probability, read from a separate call on the step's state. The separator rule takes at most
    one position per masked run while runs remain, each as far as possible from the step's other
    picks. The midpoint rule's first step is the first step of Theorem 1's schedule (lower
    midpoints of equal intervals of the single masked run), every later step the midpoint of each
    of the longest masked runs; midpoint_ties_spread spreads the picks among runs tied at the
    shortest length taken (lattice.core.selection.midpoint_indices, ties="spread").

    With chain_within each step's set is revealed one coordinate at a time, left to right within
    the set, each charged from the state holding the coordinates before it: the same schedule's
    exact chain, so a decode less its chain is the excess of decoding that schedule in parallel.
    The sets are the decode's, since no charge reads the generator.

    Records, when given: ``stats`` (a dict) gets the mean gap between consecutive selected
    positions of every step under "gaps"; ``sets`` (a list) gets every step's selected positions,
    (rows, width) on the CPU; ``timing`` (a dict) gets the seconds of every step's selection under
    "select" (the confidence rules' maximum probability included, the call it is read from
    excluded) and of the call that charges the step under "forward", the device synchronised
    around each.

    With on_device the rules select through their batch forms (lattice.core.selection), which pick
    the row forms' sets; random then draws on the device from a generator seeded by the one
    passed, the same law with other draws.
    """
    B, L = x.shape
    if widths is None:
        k = (L - prefix) // steps
        widths = [k] * (steps - 1) + [None]        # None: what is left
    cur = torch.full_like(x, MASK)
    mask = torch.ones_like(x, dtype=torch.bool)
    if prefix:
        cur[:, :prefix] = x[:, :prefix]
        mask[:, :prefix] = False
    if chain_within and timing is not None:
        raise ValueError("timing reads one charging call a step; the chain makes one per coordinate")
    if on_device and rule in ("dilated", "midpoint_ties_spread"):
        raise ValueError(f"the {rule} rule has no batch form")
    pick_gen = generator
    if on_device and rule == "random":
        pick_gen = torch.Generator(device=x.device)
        pick_gen.manual_seed(int(torch.randint(0, 2 ** 62, (1,), generator=generator)))
    order = dilated_order(L) if rule == "dilated" else None
    total, scored = 0.0, 0
    for t, w in enumerate(widths):
        left = int(mask[0].sum())
        kk = left if w is None else min(w, left)
        if kk <= 0:
            break
        conf = None
        t_sel = None
        if rule in MODEL_RULES:
            lg = forward_bf16(net, cur)
            if timing is not None:
                _sync(x)
                t_sel = time.perf_counter()
            conf = lg.float().softmax(-1).max(-1).values
        if timing is not None and t_sel is None:
            _sync(x)
            t_sel = time.perf_counter()
        if rule == "dilated":
            take = [i for i in order if bool(mask[0, i])][:kk]
            S = torch.tensor(take, device=cur.device).sort().values.unsqueeze(0).expand(B, -1)
        elif rule == "separator":
            if on_device:
                S = SEL.separator_batch(mask, kk)
            else:
                S = torch.stack([SEL.separator(mask[r], kk) for r in range(B)]).to(cur.device)
        elif rule == "midpoint":
            if on_device:
                S = SEL.midpoint_batch(mask, kk, first_step=(t == 0))
            else:
                S = torch.stack([SEL.midpoint(mask[r], kk, first_step=(t == 0))
                                 for r in range(B)]).to(cur.device)
        elif rule == "midpoint_ties_spread":
            S = torch.stack([SEL.midpoint(mask[r], kk, first_step=(t == 0), ties="spread")
                             for r in range(B)]).to(cur.device)
        else:
            S = select_sets(mask, kk, rule=rule, generator=pick_gen, conf=conf,
                            min_distance=min_distance, on_device=on_device)
        if timing is not None:
            _sync(x)
            timing.setdefault("select", []).append(time.perf_counter() - t_sel)
        if stats is not None and S.shape[1] > 1:
            stats.setdefault("gaps", []).append(float(S.sort(1).values.diff(1).float().mean()))
        if sets is not None:
            sets.append(S.cpu())
        tok = x.gather(1, S)
        if chain_within:
            for col in S.sort(1).values.split(1, dim=1):
                one = x.gather(1, col)
                pv = logits_at(net, cur, col)
                total += float(F.cross_entropy(pv.reshape(-1, V).float(), one.reshape(-1),
                                               reduction="sum"))
                cur = cur.scatter(1, col, one)
        else:
            if timing is not None:
                _sync(x)
                t_fw = time.perf_counter()
            pv = logits_at(net, cur, S)
            if timing is not None:
                _sync(x)
                timing.setdefault("forward", []).append(time.perf_counter() - t_fw)
            total += float(F.cross_entropy(pv.reshape(-1, V).float(), tok.reshape(-1),
                                           reduction="sum"))
        scored += S.numel()
        cur = cur.scatter(1, S, tok)
        mask = mask.scatter(1, S, False)
        if not bool(mask.any()):
            break
    return total / scored / LN2


@torch.no_grad()
def sliding_chain_bpc(net, split, seq=256, prefix=192, bs=64, limit=None, device="cuda"):
    """The exact chain in identity order at long context, the way the character-level literature
    evaluates: windows of ``seq`` characters overlapping so that only their last seq - prefix
    characters are scored, with everything before them revealed. ``limit`` reads only the first
    ``limit`` characters of the split."""
    tail = seq - prefix
    total, chars = 0.0, 0
    starts = np.arange(0, len(split) - seq, tail)
    if limit:
        starts = starts[: max(1, limit // tail)]
    buf = []
    for st in starts:
        buf.append(np.asarray(split[st:st + seq]))
        if len(buf) == bs or st == starts[-1]:
            x = torch.from_numpy(np.stack(buf)).long().to(device)
            xin = x.clone()
            xin[:, prefix:] = MASK
            S = torch.arange(prefix, seq, device=device).expand(x.shape[0], -1)
            tok = x.gather(1, S)
            b = chain_bpc(net, xin, S, tok)
            total += b * tok.numel()
            chars += tok.numel()
            buf = []
    return total / chars


def elbo_draw(x, seed, d, i):
    """The rates and masks of batch i of draw d, the same for every ELBO reader: a rate per row
    (a uniform clamped to [0.01, 1]) and every position masked at its row's rate, no position
    forced (a forced position would be charged the weight 1/u like the others and loosen the
    bound)."""
    u = fixed_rates(x.shape[0], seed + 977 * d + i)
    xin, m = mask_batch(x, u, torch.Generator().manual_seed(seed + 331 * d + i), at_least_one=False)
    return u, xin, m


@torch.no_grad()
def mdm_elbo_bpc(model, windows, bs=32, draws=32, seed=1234, device="cuda"):
    """The masked-diffusion bound on the given windows: over ``draws`` fixed draws of rates and
    masks (elbo_draw), every masked position's cross-entropy weighted by the reciprocal of its
    row's rate, per character of the windows."""
    seq = len(windows[0])
    total, chars = 0.0, 0
    for d in range(draws):
        buf = []
        for i, w in enumerate(windows):
            buf.append(w)
            if len(buf) == bs or i == len(windows) - 1:
                x = torch.from_numpy(np.stack(buf)).long()
                u, xin, m = elbo_draw(x, seed, d, i)
                x, xin, m = x.to(device), xin.to(device), m.to(device)
                lg = forward_bf16(model, xin)
                ce = F.cross_entropy(lg[m].float(), x[m], reduction="none")
                w8 = (1.0 / u).to(device).unsqueeze(1).expand_as(m)[m]
                total += float((w8 * ce).sum())
                chars += x.shape[0] * seq
                buf = []
    return total / chars / LN2


@torch.no_grad()
def two_stream_elbo_bpc(model, windows, bs=32, draws=32, seed=1234, device="cuda"):
    """The masked-diffusion bound as the two-stream network presents it, on the draws of
    mdm_elbo_bpc. A query at rank j reads content at ranks below j, so every revealed coordinate
    at rank zero and every masked coordinate at rank one gives each masked position the revealed
    set and nothing else: the product law the bound integrates. The network takes one rank vector
    per call, so each row is read on its own, with its own rate and mask; ``bs`` only groups rows
    for the draws."""
    seq = len(windows[0])
    total, chars = 0.0, 0
    for d in range(draws):
        buf = []
        for i, w in enumerate(windows):
            buf.append(w)
            if len(buf) == bs or i == len(windows) - 1:
                x = torch.from_numpy(np.stack(buf)).long()
                u, xin, m = elbo_draw(x, seed, d, i)
                for r in range(x.shape[0]):
                    if bool(m[r].any()):
                        mr = m[r].to(device)
                        lg = forward_bf16(model, xin[r:r + 1].to(device), mr.long())
                        ce = F.cross_entropy(lg[0, mr].float(), x[r].to(device)[mr], reduction="sum")
                        total += float(ce) / float(u[r])
                chars += x.shape[0] * seq
                buf = []
    return total / chars / LN2


@torch.no_grad()
def graded_path_bpc(net, channel, x, steps, character_at_once=False, chain_within=False):
    """Bits per character of a complete decode on a channel's levels, the spread rule choosing
    each step's set among the coordinates not yet at a character.

    Every step advances its set by one level, (length x (levels - 1)) // steps coordinate
    advances a step. On the absorbing channel an advance commits the character. On the graded
    channel the first advance reveals the character's class and is charged the model's
    probability of the true class (its character probabilities summed over the class); the second
    commits the character and is charged its cross-entropy; the total is divided by the
    characters, so both stages count. With ``character_at_once`` every advance goes straight from
    the mask to the character, length // steps a step: the graded weights revealing each
    character at once, as the absorbing channel reveals it. With ``chain_within`` each step's set
    is revealed one coordinate at a time, in position order, each charged at its stage from the
    state holding the coordinates before it: the same schedule's exact chain.
    """
    B, L = x.shape
    level = torch.zeros_like(x)
    total, scored = 0.0, 0
    per = max(1, (L if character_at_once else L * (channel.levels - 1)) // steps)
    classes = channel.classes.to(x.device)
    for _ in range(steps):
        pending = level < channel.levels - 1
        if not bool(pending.any()):
            break
        S = select_sets(pending, min(per, int(pending.sum(1).min())), rule="spread")
        if chain_within and not character_at_once:
            for m in range(S.shape[1]):
                col = S[:, m:m + 1]
                lg = forward_bf16(net, channel.render(x, level))
                lvm = level.gather(1, col)
                lp = lg.gather(1, col.unsqueeze(-1).expand(-1, -1, V)).float().log_softmax(-1)
                tok = x.gather(1, col)
                commit_m = lvm == channel.levels - 2
                total += float((-lp.gather(-1, tok.unsqueeze(-1)).squeeze(-1) * commit_m).sum())
                scored += int(commit_m.sum())
                if channel.levels > 2:
                    opens_m = lvm == 0
                    if bool(opens_m.any()):
                        same = classes.view(1, 1, -1) == classes[tok].unsqueeze(-1)
                        lpb = torch.logsumexp(lp.masked_fill(~same, float("-inf")), -1)
                        total += float((-lpb * opens_m).sum())
                level = channel.advance(level, col)
            continue
        lg = forward_bf16(net, channel.render(x, level))
        commits = (torch.ones_like(S, dtype=torch.bool) if character_at_once
                   else level.gather(1, S) == channel.levels - 2)
        if not character_at_once and channel.levels > 2:
            opens = level.gather(1, S) == 0
            if bool(opens.any()):
                lp = lg.gather(1, S.unsqueeze(-1).expand(-1, -1, V)).float().log_softmax(-1)
                same = classes.view(1, 1, -1) == classes[x.gather(1, S)].unsqueeze(-1)
                lpb = torch.logsumexp(lp.masked_fill(~same, float("-inf")), -1)
                total += float((-lpb * opens).sum())
        if bool(commits.any()):
            pv = lg.gather(1, S.unsqueeze(-1).expand(-1, -1, V))
            tok = x.gather(1, S)
            ce = F.cross_entropy(pv.reshape(-1, V).float(), tok.reshape(-1),
                                 reduction="none").view(commits.shape)
            total += float((ce * commits).sum())
            scored += int(commits.sum())
        level = channel.advance(level, S, to=(channel.levels - 1 if character_at_once else None))
    return total / max(scored, 1) / LN2
