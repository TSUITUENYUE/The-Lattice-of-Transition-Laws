"""GSM8K accuracy of LLaDA-8B-Base under one selection rule and one step budget.

    python -m lattice.experiments.llada_gsm8k --model DATA/llada/LLaDA-8B-Base --data DATA/llada/gsm8k \
        --rule confidence_min_distance --steps 8 --out DIR [--temperature 1.0] [--offset 0 --limit 0]

The released weights under the authors' single-block sampler (sampling/llada_decode.py): a
256-token answer after a five-shot prompt, greedy unless --temperature is given, with only the
selection rule changed. Each problem is decoded on its own, unpadded, and scored as lm-eval's
gsm8k task scores it (evaluation/gsm8k.py). Problem i is decoded with the seed
seed * 1000003 + i, so a shard decodes exactly what a full run decodes.

Every problem's record goes to OUT/<rule>_s<steps>[_t<temperature>]/q<index>.json as soon as it is
decoded: the few-shot indices, the generated tokens and text, both filters' outcomes, and for
every step the committed positions, whether they break either constraint, and the time of the
forward pass, of the argmax or draw, and of the selection. An interrupted run resumes at the
first problem without a record.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..evaluation.gsm8k import build_prompts, load_splits, score
from ..models.llada import environment, load_llada
from ..sampling.llada_decode import prompted
from ..sampling.llada_selection import RULES
from ..utils.records import atomic_json, check_protocol


def cell_name(rule, steps, temperature):
    return f"{rule}_s{steps}" + (f"_t{temperature:g}" if temperature > 0 else "")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the LLaDA-8B-Base directory")
    ap.add_argument("--data", required=True, help="the directory holding test.json and train.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rule", choices=RULES, default="confidence")
    ap.add_argument("--steps", type=int, default=8, help="decoding steps of the answer")
    ap.add_argument("--length", type=int, default=256, help="tokens generated after the prompt")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 decodes greedily; above 0 each committed token is drawn at this temperature")
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--min-distance", type=int, default=6,
                    help="the constrained rules' minimum distance (the decay length of the kernel "
                         "measured on these weights, experiments/llada_kernel.py)")
    ap.add_argument("--offset", type=int, default=0, help="the first problem of this shard")
    ap.add_argument("--limit", type=int, default=0, help="problems in this shard (0: to the end)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    test, train, sha = load_splits(args.data)
    prompts = build_prompts(test, train, args.shots, args.seed)
    root = Path(args.out) / cell_name(args.rule, args.steps, args.temperature)
    check_protocol(root, dict(rule=args.rule, steps=args.steps, length=args.length,
                              temperature=args.temperature, shots=args.shots, seed=args.seed,
                              min_distance=args.min_distance, test_sha256=sha["test"], train_sha256=sha["train"],
                              **environment(args.model, args.device)))
    end = min(len(test), args.offset + args.limit) if args.limit else len(test)
    pending = []
    for i in range(args.offset, end):
        path = root / f"q{i:04d}.json"
        if not path.exists():
            pending.append(i)
            continue
        record = json.loads(path.read_text())
        if record["question_index"] != i or len(record["token_ids"]) != args.length:
            raise SystemExit(f"{path} does not hold problem {i}")
    print(f"{root.name}: {len(pending)} of {end - args.offset} problems to decode", flush=True)
    if not pending:
        return
    model, tokenizer, mask_id = load_llada(args.model, args.device)
    for i in pending:
        text, shots = prompts[i]
        prompt = tokenizer(text, return_tensors="pt").input_ids[0].to(args.device)
        ids, trace, seconds = prompted(model, prompt, mask_id, args.length, args.steps, args.rule,
                                       args.seed * 1_000_003 + i, args.min_distance, args.device,
                                       args.temperature)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        record = dict(question_index=i, fewshot_indices=shots, prompt_tokens=prompt.numel(),
                      token_ids=ids, generated_text=decoded, steps=trace, seconds=seconds,
                      **score(decoded, test[i]["answer"]))
        atomic_json(root / f"q{i:04d}.json", record)
        print("QUESTION " + json.dumps(dict(index=i, strict=record["strict_match"], seconds=round(seconds, 2),
                                            relaxed_distance_steps=sum(t["breaks_min_distance"] for t in trace),
                                            relaxed_separator_steps=sum(t["breaks_separator"] for t in trace))),
              flush=True)


if __name__ == "__main__":
    main()
