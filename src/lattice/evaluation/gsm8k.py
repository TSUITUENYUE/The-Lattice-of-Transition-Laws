"""GSM8K scored as lm-eval's gsm8k task scores it: the five-shot prompt format, the stop sequences
and the two answer filters (strict match and flexible extract) with the task's normalisation.

The few-shot examples of each test problem are drawn from the full training split by one
random.Random(seed) in test order, five per problem (random.sample over the split, as lm-eval's
sampler draws); the prompts are not claimed to be byte-for-byte lm-eval's. The draw is made for
the whole test set, so a shard sees the context it would have seen in one run.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

STRICT = re.compile(r"#### (\-?[0-9\.\,]+)")
FLEX = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
UNTIL = ("Question:", "</s>", "<|im_end|>")
TEST_SIZE, TRAIN_SIZE = 1319, 7473


def load_splits(data):
    """(test, train, sha256 of each file) from DATA/test.json and DATA/train.json."""
    data = Path(data)
    files = {name: data / f"{name}.json" for name in ("test", "train")}
    test, train = (json.loads(files[n].read_text()) for n in ("test", "train"))
    if (len(test), len(train)) != (TEST_SIZE, TRAIN_SIZE):
        raise ValueError(f"expected the full splits ({TEST_SIZE}, {TRAIN_SIZE}), got "
                         f"({len(test)}, {len(train)}): run setup/prepare_llada.sh")
    sha = {n: hashlib.sha256(f.read_bytes()).hexdigest() for n, f in files.items()}
    return test, train, sha


def norm(s):
    """The task's normalisation of an answer: commas, dollar signs, everything up to '#### '
    and a final full stop removed, then stripped and lowercased."""
    s = s.strip()
    for pattern in (",", r"\$", r"(?s).*#### ", r"\.$"):
        s = re.sub(pattern, "", s)
    return s.strip().lower()


def build_prompts(test, train, shots, seed):
    """[(prompt, few-shot indices)] per test problem."""
    rng = random.Random(seed)
    out = []
    for doc in test:
        indices = rng.sample(range(len(train)), shots)
        if any(train[i]["question"] == doc["question"] for i in indices):
            raise ValueError("a few-shot example repeats its test question")
        context = "\n\n".join(f"Question: {train[i]['question']}\nAnswer: {train[i]['answer']}"
                              for i in indices)
        out.append((f"{context}\n\nQuestion: {doc['question']}\nAnswer:", indices))
    return out


def score(text, answer):
    """Both filters of the generated text against the reference answer."""
    for stop in UNTIL:
        text = text.split(stop)[0]
    target = norm(answer.split("#### ")[-1])
    strict = STRICT.search(text)
    candidates = FLEX.findall(text)
    flexible = norm([a or b for a, b in candidates][-1]) if candidates else None
    return dict(strict_match=bool(strict and norm(strict.group(1)) == target),
                flexible_match=flexible == target, target=target,
                strict_prediction=norm(strict.group(1)) if strict else None,
                flexible_prediction=flexible, scored_text=text)
