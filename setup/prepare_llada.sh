#!/usr/bin/env bash
# The LLaDA experiments' weights and data, in $DATA_ROOT/llada/:
#   LLaDA-8B-Base/        GSAI-ML/LLaDA-8B-Base from the Hugging Face hub, with its modelling code (about 16 GB)
#   gpt2-large/           openai-community/gpt2-large, the generative-perplexity scorer (about 3 GB)
#   gsm8k/{test,train}.json  the full GSM8K splits (1,319 and 7,473 problems) of openai/grade-school-math
#   HumanEval.jsonl.gz    openai/human-eval's 164 problems
#   wikitext103_test.txt  the first 1,200 rows of WikiText-103's raw test split, joined: the held-out text
#                         the kernel is read on (357,265 characters)
# The GSM8K and HumanEval files are checked against the sha256 sums of the copies behind the paper's
# numbers, and the WikiText text against its length. The hub revisions of the paper's model copies were not
# recorded; the revisions downloaded here are written to REVISIONS. Needs huggingface_hub and pyarrow.
source "$(dirname "$0")/../scripts/_common.sh"
D="$DATA_ROOT/llada"
mkdir -p "$D/gsm8k"
G=https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data
for s in train test; do
  [ -f "$D/gsm8k/$s.jsonl" ] || curl -fsSL -o "$D/gsm8k/$s.jsonl" "$G/$s.jsonl"
done
[ -f "$D/HumanEval.jsonl.gz" ] || curl -fsSL -o "$D/HumanEval.jsonl.gz" \
  https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz
"$PYTHON" - "$D" <<'PY'
import hashlib, json, sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

D = Path(sys.argv[1])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
want = {"gsm8k/train.jsonl": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
        "gsm8k/test.jsonl": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
        "HumanEval.jsonl.gz": "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"}
for name, digest in want.items():
    assert sha(D / name) == digest, f"{name}: sha256 {sha(D / name)}, expected {digest}"
# the harness reads each split as one JSON list, written exactly as the paper's copies were
for split, digest in (("train", "4c6509bb0835be228e68ee0e3ca8f8bf82dac0053da04cae3fef9e345f50fb5a"),
                      ("test", "179294d0f1f2b81c884f49262344c3807f834795f96adbce5c79706150950159")):
    rows = [json.loads(line) for line in open(D / "gsm8k" / f"{split}.jsonl") if line.strip()]
    (D / "gsm8k" / f"{split}.json").write_text(json.dumps(rows, ensure_ascii=False) + "\n")
    assert sha(D / "gsm8k" / f"{split}.json") == digest, f"gsm8k/{split}.json differs from the paper's copy"
if not (D / "wikitext103_test.txt").exists():
    import pyarrow.parquet as pq
    parquet = hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/test-00000-of-00001.parquet",
                              repo_type="dataset")
    text = "".join(pq.read_table(parquet).column("text").to_pylist()[:1200])
    assert len(text) == 357265, f"WikiText text of {len(text)} characters, expected 357,265"
    (D / "wikitext103_test.txt").write_text(text)
revisions = {}
for repo, folder, patterns in (("GSAI-ML/LLaDA-8B-Base", "LLaDA-8B-Base", None),
                               ("openai-community/gpt2-large", "gpt2-large", ["*.json", "*.txt", "*.safetensors"])):
    revisions[repo] = HfApi().model_info(repo).sha
    snapshot_download(repo, revision=revisions[repo], local_dir=D / folder, allow_patterns=patterns)
(D / "REVISIONS").write_text("".join(f"{repo} {rev}\n" for repo, rev in revisions.items()))
print("LLaDA data ready; checksums match the paper's copies; revisions in", D / "REVISIONS")
PY
