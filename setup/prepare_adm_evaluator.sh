#!/usr/bin/env bash
# ADM's ImageNet evaluation suite (guided-diffusion, evaluations/evaluator.py), its Inception graph
# and the VIRTUAL_imagenet256_labeled.npz reference batch (about 2 GB), checked against the md5
# sums of the copies the paper's reads used, and a TensorFlow environment of their own
# (TensorFlow 2.15.1; numpy 1.23.5, since the evaluator uses np.bool). Every ImageNet FID is read
# with it, on the CPU. Output: $DATA_ROOT/adm_evaluator/{evaluator.py,
# classify_image_graph_def.pb,VIRTUAL_imagenet256_labeled.npz,venv/}.
source "$(dirname "$0")/../scripts/_common.sh"
D="$DATA_ROOT/adm_evaluator"
REF=https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches
mkdir -p "$D"
cd "$D"
# fetch FILE URL: download FILE unless it is there, through a partial file
fetch () {
  [ -f "$1" ] && return 0
  curl -fL --retry 3 -o "$1.part" "$2"
  mv "$1.part" "$1"
}
fetch evaluator.py https://raw.githubusercontent.com/openai/guided-diffusion/main/evaluations/evaluator.py
fetch classify_image_graph_def.pb "$REF/classify_image_graph_def.pb"
fetch VIRTUAL_imagenet256_labeled.npz "$REF/imagenet/256/VIRTUAL_imagenet256_labeled.npz"
"$PYTHON" - <<'PY'
import hashlib
expected = {"evaluator.py": "273848f1ef73023a9c467577f294b1c2",
            "classify_image_graph_def.pb": "e6bb154e85f5d4331c22abcddb5dcf31",
            "VIRTUAL_imagenet256_labeled.npz": "14c5913e1d24e8827c1ba61d8fc5aeb9"}
for name, md5 in expected.items():
    h = hashlib.md5()
    with open(name, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != md5:
        raise SystemExit(f"{name}: md5 {h.hexdigest()}, the paper's copy has {md5}")
print("evaluator, Inception graph and reference batch match the paper's copies")
PY
[ -x venv/bin/python ] || "${PYTHON_FOR_VENV:-python3.11}" -m venv venv
venv/bin/pip install -q "tensorflow==2.15.1" "numpy==1.23.5" "scipy==1.11.4" requests tqdm
echo "ADM evaluation suite ready in $D"
