"""Reveal orders for MAR's masked decoder on its 16 x 16 token grid.

MAR reveals a permutation of the 256 tokens in chunks whose sizes follow its own cosine schedule;
the permutation is the only thing a rule chooses, so two rules at one step count differ in which
tokens each step reveals and in nothing else. The rules:

    raster     row by row
    random     a uniformly random permutation, MAR's own
    spread     the bit-reversed Morton order, so every stretch of the order is spread evenly over
               the grid; a random reflection of rows and of columns breaks ties between images
    midpoint   farthest point first: each next token is the one farthest (Chebyshev distance)
               from the tokens already revealed, ties broken at random (on a line, the midpoint
               of a masked run)
    low_discrepancy
               the low-discrepancy rule (Besnier et al.), a Halton order: the 2-D Halton sequence
               in bases 2 (rows) and 3 (columns), each cell taken the first time the sequence hits
               it, from a random offset into the sequence
    nested     nested dissection: the middle row and column of the grid first, then the middle
               row and column of each of the four parts they leave, and so on down to single
               tokens; within one level the tokens follow the spread order, so the tokens of
               one level are spread over the grid
    spread_raster
               spread for the first quarter of the grid, then row by row: the 64 tokens the spread
               rule reveals first, then every other token in row-major order. Its early steps are
               the spread rule's and its late steps are contiguous, so a floor that charges all of
               MAR's steps alike and one that charges only its early steps rank it on opposite
               sides of random

MAR keeps the first entries of a permutation masked longest (mask_by_order in its models/mar.py),
so it reveals a permutation from its end. midpoint_order, low_discrepancy_order, nested_order and
spread_raster_order are built with the first token to reveal first, and install_rule reverses them
before MAR reads them. Raster, random and
spread are passed as built: MAR then reveals the raster order from the bottom-right corner, the
reverse of a uniformly random permutation is again uniformly random, and the reverse of the spread
order is its 180-degree rotation, which its random reflections already produce.
"""
import math

import numpy as np
import torch

RULES = ("random", "raster", "spread", "midpoint", "low_discrepancy", "nested", "spread_raster")
REVERSED = ("midpoint", "low_discrepancy", "nested", "spread_raster")    # built first-revealed-first, reversed for MAR


def _bitrev(v, bits):
    r = 0
    for _ in range(bits):
        r = (r << 1) | (v & 1)
        v >>= 1
    return r


def raster_order(side):
    return np.arange(side * side)


def spread_order(side, rng):
    bits = int(math.log2(side))
    idx = []
    for k in range(side * side):
        m = _bitrev(k, 2 * bits)          # bit-reversed Morton index
        r = c = 0
        for b in range(bits):             # de-interleave into row and column
            r |= ((m >> (2 * b + 1)) & 1) << b
            c |= ((m >> (2 * b)) & 1) << b
        idx.append(r * side + c)
    idx = np.array(idx)
    if rng.random() < 0.5:                # reflect rows
        idx = (side - 1 - idx // side) * side + idx % side
    if rng.random() < 0.5:                # reflect columns
        idx = (idx // side) * side + (side - 1 - idx % side)
    return idx


def _radical_inverse(i, base):
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def low_discrepancy_order(side, rng):
    """The low-discrepancy order: the cells in the order a 2-D Halton sequence first reaches them."""
    n = side * side
    seen, order = set(), []
    i = 1 + int(rng.integers(0, 997))
    while len(order) < n and i < 200000:
        r = min(side - 1, int(_radical_inverse(i, 2) * side))
        c = min(side - 1, int(_radical_inverse(i, 3) * side))
        k = r * side + c
        if k not in seen:
            seen.add(k)
            order.append(k)
        i += 1
    for k in range(n):                    # cells the sequence never reached, in raster order
        if k not in seen:
            order.append(k)
    return np.array(order)


def midpoint_order(side, rng):
    n = side * side
    rr, cc = np.divmod(np.arange(n), side)
    start = rng.integers(n)
    order = [start]
    dist = np.maximum(np.abs(rr - rr[start]), np.abs(cc - cc[start])).astype(float)
    dist[start] = -1
    for _ in range(n - 1):
        far = np.flatnonzero(dist == dist.max())
        nxt = int(rng.choice(far))
        order.append(nxt)
        d = np.maximum(np.abs(rr - rr[nxt]), np.abs(cc - cc[nxt]))
        dist = np.minimum(dist, d)
        dist[nxt] = -1
    return np.array(order)


def dissection_levels(side):
    """The nested-dissection level of every token: 0 on the middle row and column of the grid, 1 on
    those of the parts they leave, and so on (a part one token wide is cut at its middle token)."""
    level = np.full((side, side), -1)

    def cut(r0, r1, c0, c1, lv):
        if r0 >= r1 or c0 >= c1:
            return
        rm, cm = (r0 + r1) // 2, (c0 + c1) // 2
        if r1 - r0 > 1:
            level[rm, c0:c1] = lv
        if c1 - c0 > 1:
            level[r0:r1, cm] = lv
        if r1 - r0 == 1 and c1 - c0 == 1:
            level[r0, c0] = lv
            return
        rows = [(r0, rm), (rm + 1, r1)] if r1 - r0 > 1 else [(r0, r1)]
        cols = [(c0, cm), (cm + 1, c1)] if c1 - c0 > 1 else [(c0, c1)]
        for a, b in rows:
            for c, d in cols:
                cut(a, b, c, d, lv + 1)

    cut(0, side, 0, side, 0)
    assert (level >= 0).all()
    return level.ravel()


def nested_order(side, rng):
    rank = np.empty(side * side, dtype=int)
    rank[spread_order(side, rng)] = np.arange(side * side)
    return np.lexsort((rank, dissection_levels(side)))


def spread_raster_order(side, rng):
    """Spread for the first quarter of the grid, then row by row, first revealed first: the tokens MAR
    reveals first under the spread rule (the spread order read from its end), then the rest ascending."""
    first = spread_order(side, rng)[::-1][: side * side // 4]
    rest = np.setdiff1d(np.arange(side * side), first)
    return np.concatenate([first, rest])


def mar_chunks(num_iter, seq_len=256):
    """How many tokens each of MAR's steps reveals: its cosine schedule (models/mar.py sample_tokens)."""
    sizes, masked = [], seq_len
    for step in range(num_iter):
        if step == num_iter - 1:
            sizes.append(masked)
            break
        m = max(1, min(masked - 1, math.floor(seq_len * math.cos(math.pi / 2 * (step + 1) / num_iter))))
        sizes.append(masked - m)
        masked = m
    return sizes


def sequence_floor(order, kernel, num_iter, side=16):
    """The first-order floor of one reveal order (first revealed first) at ``num_iter`` steps: the
    kernel at the Chebyshev distance of every two tokens adjacent in the order within one of MAR's
    steps, summed; ``kernel`` maps distances 1 .. D to values, and farther pairs take the value at D."""
    far = kernel[max(kernel)]
    tot, i = 0.0, 0
    for n in mar_chunks(num_iter, side * side):
        seq = [int(x) for x in order[i:i + n]]
        i += n
        for a, b in zip(seq, seq[1:]):
            tot += kernel.get(max(abs(a // side - b // side), abs(a % side - b % side)), far)
    return tot


def orders_for(rule, bsz, side, seed):
    """(bsz, side * side) orders of one rule as built (before any reversal), from one numpy generator."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(bsz):
        if rule == "random":
            o = rng.permutation(side * side)
        elif rule == "raster":
            o = raster_order(side)
        elif rule == "spread":
            o = spread_order(side, rng)
        elif rule == "midpoint":
            o = midpoint_order(side, rng)
        elif rule == "low_discrepancy":
            o = low_discrepancy_order(side, rng)
        elif rule == "nested":
            o = nested_order(side, rng)
        elif rule == "spread_raster":
            o = spread_raster_order(side, rng)
        else:
            raise ValueError(rule)
        out.append(o)
    return torch.tensor(np.stack(out), dtype=torch.long)


def install_rule(model, rule, seed, device):
    """Replace MAR's random permutation by the rule's, drawn for the n-th batch from seed + n.
    Returns the batch counter, which a resumed run restores."""
    state = {"n": 0}

    def sample_orders(bsz):
        o = orders_for(rule, bsz, int(math.isqrt(model.seq_len)), seed + state["n"])
        if rule in REVERSED:
            o = o.flip(1)
        state["n"] += 1
        return o.to(device)

    model.sample_orders = sample_orders
    return state
