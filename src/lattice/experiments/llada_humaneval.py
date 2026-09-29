"""HumanEval pass@1 of LLaDA-8B-Base under one selection rule and one step budget.

    python -m lattice.experiments.llada_humaneval generate --model DATA/llada/LLaDA-8B-Base \
        --data DATA/llada/HumanEval.jsonl.gz --rule confidence_separator --steps 8 --out DIR
    python -m lattice.experiments.llada_humaneval check-sandbox --data FILE --out DIR --sandbox bwrap
    python -m lattice.experiments.llada_humaneval evaluate --data FILE --rule R --steps S --out DIR --sandbox bwrap

generate: the zero-shot prompt alone, decoded by the GSM8K sampler (greedy, 256 tokens, one problem
at a time, unpadded; sampling/llada_decode.py) with only the selection rule changed, and cut to a
completion as lm-eval's humaneval task cuts it (evaluation/humaneval.py). Problem i is decoded
with the seed seed * 1000003 + i. Nothing is executed. Records go to OUT/<rule>_s<steps>/q<i>.json
as soon as each problem is decoded, and an interrupted run resumes at the first problem without one.

check-sandbox: checks the sandbox (no network, read-only root and home, private /tmp, every
canonical solution passing, every empty body failing, an endless loop timing out) and writes
OUT/sandbox_check.json. evaluate refuses to run without a check of the same sandbox and data.

evaluate: runs each problem's program in a fresh sandbox and writes its outcome to
OUT/<rule>_s<steps>/scores/q<i>.json; it resumes too. A failure of the sandbox stops the run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ..evaluation.humaneval import (RUNTIMES, TASKS, check_sandbox, completion_of, data_sha256,
                                    load_problems, program_of, run_in_sandbox, verify_sandbox)
from ..models.llada import environment, load_llada
from ..sampling.llada_decode import prompted
from ..sampling.llada_selection import RULES
from ..utils.records import atomic_json, check_protocol


def generate(model, tokenizer, mask_id, problems, root, rule, steps, length, seed, min_distance, device):
    """Decode every problem of `problems` (index, problem) without a record in root."""
    for i, problem in problems:
        path = root / f"q{i:04d}.json"
        if path.exists():
            continue
        prompt = tokenizer(problem["prompt"], return_tensors="pt").input_ids[0].to(device)
        ids, trace, seconds = prompted(model, prompt, mask_id, length, steps, rule,
                                       seed * 1_000_003 + i, min_distance, device)
        atomic_json(path, dict(index=i, task_id=problem["task_id"], prompt=problem["prompt"],
                               prompt_token_ids=prompt.cpu().tolist(), token_ids=ids,
                               raw_text=tokenizer.decode(ids, skip_special_tokens=False),
                               completion=completion_of(tokenizer, ids), steps=trace, seconds=seconds))
        print(f"HUMANEVAL {root.name} {i + 1}/{TASKS} {problem['task_id']}", flush=True)


def evaluate(problems, root, verified):
    """Run every program of a cell in the sandbox; returns the number that passed."""
    passed = 0
    for i, problem in enumerate(problems):
        generation = json.loads((root / f"q{i:04d}.json").read_text())
        if generation["task_id"] != problem["task_id"]:
            raise SystemExit(f"{root}/q{i:04d}.json holds {generation['task_id']}, not {problem['task_id']}")
        program = program_of(problem, generation["completion"])
        program_sha256 = hashlib.sha256(program.encode()).hexdigest()
        path = root / "scores" / f"q{i:04d}.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["program_sha256"] != program_sha256 or result["sandbox"] != verified:
                raise SystemExit(f"{path} scored another program or sandbox")
        else:
            result = dict(index=i, task_id=problem["task_id"], program_sha256=program_sha256,
                          sandbox=verified, **run_in_sandbox(program, verified))
            atomic_json(path, result)
        passed += result["status"] == "passed"
    print(f"HUMANEVAL {root.name}: {passed}/{len(problems)} passed", flush=True)
    return passed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["generate", "check-sandbox", "evaluate"])
    ap.add_argument("--data", required=True, help="HumanEval.jsonl.gz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", help="the LLaDA-8B-Base directory (generate)")
    ap.add_argument("--rule", choices=RULES, default="confidence")
    ap.add_argument("--steps", type=int, default=8, help="decoding steps of the completion")
    ap.add_argument("--length", type=int, default=256, help="tokens generated after the prompt")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--min-distance", type=int, default=6, help="the constrained rules' minimum distance")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sandbox", choices=RUNTIMES, help="the runtime candidates execute in")
    ap.add_argument("--image", help="docker and podman: the local image, named by its sha256 digest")
    args = ap.parse_args(argv)
    problems = load_problems(args.data)
    out = Path(args.out)
    root = out / f"{args.rule}_s{args.steps}"
    if args.stage == "generate":
        if not args.model:
            ap.error("generate needs --model")
        check_protocol(root, dict(rule=args.rule, steps=args.steps, length=args.length, seed=args.seed,
                                  min_distance=args.min_distance, data_sha256=data_sha256(args.data),
                                  **environment(args.model, args.device)))
        todo = [(i, p) for i, p in enumerate(problems) if not (root / f"q{i:04d}.json").exists()]
        print(f"{root.name}: {len(todo)} of {TASKS} problems to decode", flush=True)
        if todo:
            model, tokenizer, mask_id = load_llada(args.model, args.device)
            generate(model, tokenizer, mask_id, todo, root, args.rule, args.steps, args.length,
                     args.seed, args.min_distance, args.device)
        return
    if not args.sandbox:
        ap.error(f"{args.stage} needs --sandbox")
    verified = verify_sandbox(args.sandbox, args.image)
    if args.stage == "check-sandbox":
        record = check_sandbox(problems, verified)
        atomic_json(out / "sandbox_check.json", dict(record, data_sha256=data_sha256(args.data)))
        print("SANDBOX_CHECKED " + json.dumps(record["probes"]), flush=True)
        return
    check_path = out / "sandbox_check.json"
    if not check_path.exists():
        raise SystemExit(f"no {check_path}: run check-sandbox with this sandbox first")
    check = json.loads(check_path.read_text())
    if check["sandbox"] != verified or check["data_sha256"] != data_sha256(args.data):
        raise SystemExit(f"{check_path} checked another sandbox or data file; run check-sandbox again")
    evaluate(problems, root, verified)


if __name__ == "__main__":
    main()
