"""Generative perplexity: a generated text re-tokenised for a causal scorer and scored in full.

Every token after the first is scored exactly once, with as much preceding text as the scorer's
context holds: the text is read in windows whose scored parts are consecutive stretches of
`stride` tokens, each window reaching back up to the context length. The first token has no
preceding context and is not scored; no beginning-of-text token is added. The loss is a float32
cross-entropy summed over the scored tokens.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def scoring_windows(length, context, stride):
    """(begin, first scored, end) for each window over a text of `length` tokens."""
    if not 1 <= stride < context:
        raise ValueError("the stride must be positive and smaller than the scorer's context")
    for target_start in range(1, length, stride):
        end = min(length, target_start + stride)
        begin = max(0, end - context)
        yield begin, target_start, end


@torch.inference_mode()
def score_ids(scorer, ids, context, stride, device):
    """Summed negative log-likelihood (nats) of ids[1:] under the scorer, with the windows."""
    ids = torch.as_tensor(ids, dtype=torch.long)
    nll, count, windows = 0.0, 0, []
    for begin, target_start, end in scoring_windows(len(ids), context, stride):
        output = scorer(ids[None, begin:end].to(device))
        logits = output.logits[0, target_start - begin - 1:end - begin - 1].float()
        nll += float(F.cross_entropy(logits, ids[target_start:end].to(device), reduction="sum"))
        count += end - target_start
        windows.append(dict(context_start=begin, target_start=target_start, target_end=end))
    if count != max(len(ids) - 1, 0):
        raise RuntimeError("the windows did not score every token after the first once")
    return dict(nll=nll, targets=count, retokenized_length=len(ids), windows=windows)


def unigram_entropy(ids):
    """Entropy in bits of the empirical distribution of the ids a sample holds."""
    counts = torch.bincount(torch.as_tensor(ids, dtype=torch.long)).double()
    freq = counts[counts > 0] / counts.sum()
    return float(-(freq * freq.log2()).sum())
