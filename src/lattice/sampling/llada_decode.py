"""The two decodes on LLaDA-8B-Base, with the selection rule as the only choice the paper varies.

prompted() is the authors' single-block sampler after a prompt (GSM8K, HumanEval, the timing of
the selection): generation length L split over `steps` steps as evenly as the authors split it,
each step committing the positions the rule selects. Greedy decoding commits each selected
position's most likely token and ranks positions by that token's probability, computed in the
model's own precision (bfloat16) as the authors' sampler computes it. With a temperature above
zero it commits a token drawn at that temperature (the mask token excluded) by inverting the
cumulative distribution with one uniform per position drawn on the CPU, so the draw does not
depend on the GPU, and ranks by the drawn token's untempered probability. The rules see the
generation block only: the prompt does not count as a revealed position. Each question is decoded
on its own, so there is no padding in any context.

unconditional() writes a sequence from all-masked states (generative perplexity): every step
commits `width` positions, each drawn from the model's distribution after nucleus truncation, and
the confidence rules rank by the probability of the most likely token, the mask token excluded.
"""
from __future__ import annotations

import time

import torch

from ..models.llada import logits as model_logits, sync
from .llada_selection import CONFIDENCE_RULES, constraint_diagnostics, pick


def transfer_counts(length, steps):
    """The authors' split of a block's positions over its steps: equal parts, the first steps
    taking one more when the length does not divide."""
    return [length // steps + int(i < length % steps) for i in range(steps)]


def sample_tokens(logits, mask_id, temperature, sampler):
    """(tokens, logits32): one token per position drawn at `temperature` with the mask token
    excluded, by inverting the CDF with a uniform from the CPU generator `sampler`."""
    blk = logits.float()
    blk[..., mask_id] = float("-inf")
    cdf = torch.softmax(blk if temperature == 1.0 else blk / temperature, dim=-1)
    cdf.cumsum_(dim=-1)
    u = torch.rand(cdf.shape[:-1], generator=sampler).to(cdf.device)
    u = (u * cdf[..., -1]).unsqueeze(-1)
    tok = torch.searchsorted(cdf, u).squeeze(-1).clamp_(max=cdf.shape[-1] - 1)
    return tok, blk


def sampled_confidence(logits32, tokens):
    """The untempered probability of each drawn token, from the float32 logits sample_tokens
    returned (mask token excluded)."""
    lse = torch.logsumexp(logits32, dim=-1, keepdim=True)
    return (logits32.gather(-1, tokens.unsqueeze(-1)) - lse).squeeze(-1).exp()


def greedy_confidence(logits, tokens):
    """The probability of each greedy token, softmax in the logits' own precision."""
    return logits.softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1).float()


@torch.inference_mode()
def prompted(model, prompt, mask_id, length, steps, rule, seed, min_distance, device, temperature=0.0):
    """Decode `length` positions after the 1-D token tensor `prompt` in `steps` steps.

    Returns (token ids, trace, seconds). The trace holds one entry per step: the committed
    positions (in the block), the step's quota, whether the committed positions break either
    constraint, and the time of the step's forward pass, of the tokens every rule needs (argmax
    or draw), and of the selection (the confidence, for the rules that read it, and the rule).
    The device is synchronised around each timed region. `seed` seeds the random rule's
    generator on the device and, with a temperature, the CPU generator of the draws.
    """
    p = prompt.numel()
    x = torch.full((1, p + length), mask_id, dtype=torch.long, device=device)
    x[0, :p] = prompt
    gen = torch.Generator(device=device).manual_seed(seed)
    sampler = torch.Generator().manual_seed(seed) if temperature > 0 else None
    counts = transfer_counts(length, steps)
    trace = []
    sync(device)
    started = time.perf_counter()
    for step, quota in enumerate(counts):
        if quota == 0:
            continue
        masked = x[:, p:].eq(mask_id)
        if int(masked.sum()) != length - sum(counts[:step]):
            raise RuntimeError("a position committed earlier is masked again")
        sync(device)
        t0 = time.perf_counter()
        logits = model_logits(model, x)[:, p:]
        sync(device)
        t1 = time.perf_counter()
        logits32 = None
        if temperature > 0:
            tokens, logits32 = sample_tokens(logits, mask_id, temperature, sampler)
        else:
            tokens = logits.argmax(-1)
        sync(device)
        t2 = time.perf_counter()
        conf = None
        if rule in CONFIDENCE_RULES:
            conf = (sampled_confidence(logits32, tokens) if temperature > 0
                    else greedy_confidence(logits, tokens))
        indices, keep = pick(rule, masked, quota, conf, gen, min_distance=min_distance)
        sync(device)
        t3 = time.perf_counter()
        selected = indices[0, keep[0]]
        if (selected.numel() != quota or selected.unique().numel() != quota
                or not bool(masked[0, selected].all())):
            raise RuntimeError(f"{rule} did not commit {quota} distinct masked positions")
        if bool(tokens[0, selected].eq(mask_id).any()):
            raise RuntimeError("the model proposed the mask token at a selected position; "
                               "the sample would be unfinished")
        x[0, p + selected] = tokens[0, selected]
        trace.append(dict(step=step, quota=quota, positions=selected.cpu().tolist(),
                          forward_seconds=t1 - t0, score_seconds=t2 - t1, select_seconds=t3 - t2,
                          **constraint_diagnostics(masked, indices, keep, min_distance)[0]))
        del logits, logits32, tokens, conf
    sync(device)
    if bool(x[:, p:].eq(mask_id).any()):
        raise RuntimeError("the decode ended with masked positions")
    return x[0, p:].cpu().tolist(), trace, time.perf_counter() - started


def nucleus(p, top_p):
    """The distributions p truncated to the smallest set of tokens whose mass reaches top_p."""
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if top_p == 1:
        return p
    sorted_p, indices = p.sort(-1, descending=True)
    keep = sorted_p.cumsum(-1) - sorted_p <= top_p
    keep[..., 0] = True
    out = torch.zeros_like(p).scatter_(-1, indices, sorted_p * keep)
    return out / out.sum(-1, keepdim=True)


def probability_confidence(logits):
    """The probability of the most likely token at each position (not its unnormalised logit)."""
    values = logits.float()
    return (values.amax(-1) - values.logsumexp(-1)).exp()


@torch.inference_mode()
def unconditional(model, mask_id, length, width, rule, batch, device, generator, top_p, min_distance):
    """Write `batch` sequences of `length` tokens from all-masked states, `width` per step.

    Returns (token ids per row, trace per row, seconds). `generator` (on the device) drives both
    the random rule and the draws, so a run continues one stream through all its cells.
    """
    if width < 1 or length < 1 or batch < 1 or length % width:
        raise ValueError("the width must divide the length, so every step commits the same number")
    x = torch.full((batch, length), mask_id, dtype=torch.long, device=device)
    traces = [[] for _ in range(batch)]
    sync(device)
    started = time.perf_counter()
    for step in range(length // width):
        masked = x.eq(mask_id)
        before = time.perf_counter()
        logits = model_logits(model, x)
        logits[:, :, mask_id] = torch.finfo(logits.dtype).min
        sync(device)
        forward = time.perf_counter() - before
        conf = probability_confidence(logits)
        indices, keep = pick(rule, masked, width, conf, generator, min_distance=min_distance)
        if (not bool(keep.all()) or any(row.unique().numel() != width for row in indices)
                or not bool(masked.gather(1, indices).all())):
            raise RuntimeError(f"{rule} did not commit {width} distinct masked positions")
        selected = logits.gather(1, indices[:, :, None].expand(batch, width, logits.shape[-1]))
        probabilities = nucleus(selected.float().softmax(-1).reshape(batch * width, -1), top_p)
        draw = torch.multinomial(probabilities, 1, generator=generator).reshape(batch, width)
        if bool(draw.eq(mask_id).any()):
            raise RuntimeError("the mask token was drawn")
        diagnostics = constraint_diagnostics(masked, indices, keep, min_distance)
        x.scatter_(1, indices, draw)
        for i in range(batch):
            traces[i].append(dict(step=step, positions=indices[i].cpu().tolist(), quota=width,
                                  forward_seconds=forward, **diagnostics[i]))
        del logits, selected, probabilities, conf, draw
    sync(device)
    if bool(x.eq(mask_id).any()):
        raise RuntimeError("the decode ended with masked positions")
    return x.cpu().tolist(), traces, time.perf_counter() - started
