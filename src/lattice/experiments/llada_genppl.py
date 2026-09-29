"""Generative perplexity of LLaDA-8B-Base samples under each selection rule, scored by GPT-2-large.

    python -m lattice.experiments.llada_genppl generate --model DATA/llada/LLaDA-8B-Base --out DIR --seed 0
    python -m lattice.experiments.llada_genppl score --scorer DATA/llada/gpt2-large --out DIR --seed 0

generate: for every width (tokens committed per step: 64, 32, 16, 8, i.e. 8 to 64 steps for 512
tokens) and every rule, 64 unconditional samples of 512 tokens in batches of 2, each token drawn
after nucleus truncation at 0.9 (sampling/llada_decode.py). One generator on the device, seeded
once, runs through the widths, the rules and the batches in that order, so each seed is one
random stream. Each batch is written whole to OUT/seed<S>/<rule>_w<width>/b<start>.json with its
token ids, texts, per-step committed positions and constraint diagnostics, and the generator's
state before and after it; an interrupted run checks that chain and resumes at the first
missing batch.

score: a separate process (LLaDA is not in memory) re-tokenises each sample's complete text for
the scorer, without truncation, and scores every token after the first once with a sliding
window at the scorer's own context (evaluation/generative_perplexity.py). One file per sample in
OUT/seed<S>/scores/<rule>_w<width>/s<index>.json holds the scorer's token ids, the summed
negative log-likelihood, the number of scored tokens and the unigram entropy of the sample's
LLaDA tokens. The displays pool these over seeds (experiments/tables_released.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ..evaluation.generative_perplexity import score_ids, unigram_entropy
from ..models.llada import environment, load_llada, load_scorer
from ..sampling.llada_decode import unconditional
from ..sampling.llada_selection import RULES
from ..utils.records import atomic_json, check_protocol

GENERATION_RULES = ("random", "confidence", "confidence_min_distance", "spread", "midpoint")
WIDTHS = (64, 32, 16, 8)


def generate_records(model, tokenizer, mask_id, root, protocol, device):
    """Write every missing batch of a seed, checking the random stream of those already written."""
    root = Path(root)
    generator = torch.Generator(device=device).manual_seed(protocol["seed"])
    for width in protocol["widths"]:
        for rule in protocol["rules"]:
            for start in range(0, protocol["samples"], protocol["batch"]):
                size = min(protocol["batch"], protocol["samples"] - start)
                path = root / f"{rule}_w{width}" / f"b{start:04d}.json"
                before = generator.get_state().cpu().tolist()
                if path.exists():
                    old = json.loads(path.read_text())
                    if old["start"] != start or old["rng_before"] != before or len(old["samples"]) != size:
                        raise SystemExit(f"{path} does not continue this seed's random stream")
                    generator.set_state(torch.tensor(old["rng_after"], dtype=torch.uint8))
                    continue
                ids, traces, seconds = unconditional(model, mask_id, protocol["length"], width, rule, size,
                                                     device, generator, protocol["nucleus"], protocol["min_distance"])
                samples = [dict(index=start + i, token_ids=seq,
                                text=tokenizer.decode(seq, skip_special_tokens=True),
                                raw_text=tokenizer.decode(seq, skip_special_tokens=False), trace=traces[i])
                           for i, seq in enumerate(ids)]
                atomic_json(path, dict(start=start, rule=rule, width=width, calls=protocol["length"] // width,
                                       samples=samples, seconds=seconds, rng_before=before,
                                       rng_after=generator.get_state().cpu().tolist()))
                print(f"GENERATED {rule} width={width} samples={start + size}/{protocol['samples']}", flush=True)


def score_records(scorer, tokenizer, root, protocol, context, stride, device):
    """Score every sample of a seed that has no score yet."""
    root = Path(root)
    for width in protocol["widths"]:
        for rule in protocol["rules"]:
            for start in range(0, protocol["samples"], protocol["batch"]):
                batch = json.loads((root / f"{rule}_w{width}" / f"b{start:04d}.json").read_text())
                for sample in batch["samples"]:
                    path = root / "scores" / f"{rule}_w{width}" / f"s{sample['index']:04d}.json"
                    sample_sha256 = hashlib.sha256(
                        json.dumps([sample["text"], sample["token_ids"]]).encode()).hexdigest()
                    if path.exists():
                        if json.loads(path.read_text())["sample_sha256"] != sample_sha256:
                            raise SystemExit(f"{path} scored another sample")
                        continue
                    ids = tokenizer(sample["text"], add_special_tokens=False, truncation=False)["input_ids"]
                    atomic_json(path, dict(index=sample["index"], sample_sha256=sample_sha256, scorer_token_ids=ids,
                                           entropy_bits=unigram_entropy(sample["token_ids"]),
                                           **score_ids(scorer, ids, context, stride, device)))
            print(f"SCORED {rule} width={width}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["generate", "score"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", help="the LLaDA-8B-Base directory (generate)")
    ap.add_argument("--scorer", help="the GPT-2-large directory (score)")
    ap.add_argument("--rules", default=",".join(GENERATION_RULES))
    ap.add_argument("--widths", default=",".join(map(str, WIDTHS)), help="tokens committed per step")
    ap.add_argument("--samples", type=int, default=64, help="samples per rule and width")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--length", type=int, default=512, help="tokens per sample")
    ap.add_argument("--nucleus", type=float, default=0.9, help="nucleus probability of the draws")
    ap.add_argument("--min-distance", type=int, default=6,
                    help="the minimum distance, the decay length of the kernel on these weights")
    ap.add_argument("--stride", type=int, default=512, help="tokens scored per window")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    root = Path(args.out) / f"seed{args.seed}"
    if args.stage == "generate":
        if not args.model:
            ap.error("generate needs --model")
        rules = args.rules.split(",")
        widths = [int(w) for w in args.widths.split(",")]
        if not set(rules) <= set(RULES) or any(w < 1 or args.length % w for w in widths):
            ap.error("unknown rule, or a width that does not divide the length")
        protocol = dict(rules=rules, widths=widths, samples=args.samples, batch=args.batch, length=args.length,
                        seed=args.seed, min_distance=args.min_distance, nucleus=args.nucleus,
                        **environment(args.model, args.device))
        check_protocol(root, protocol)
        missing = [1 for w in widths for r in rules for s in range(0, args.samples, args.batch)
                   if not (root / f"{r}_w{w}" / f"b{s:04d}.json").exists()]
        model = tokenizer = mask_id = None
        if missing:
            model, tokenizer, mask_id = load_llada(args.model, args.device)
        generate_records(model, tokenizer, mask_id, root, protocol, args.device)
        return
    if not args.scorer:
        ap.error("score needs --scorer")
    protocol = json.loads((root / "protocol.json").read_text())
    scorer, tokenizer, context = load_scorer(args.scorer, args.device)
    check_protocol(root / "scores", dict(context=context, stride=args.stride,
                                         **environment(args.scorer, args.device)))
    score_records(scorer, tokenizer, root, protocol, context, args.stride, args.device)


if __name__ == "__main__":
    main()
