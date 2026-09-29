#!/usr/bin/env bash
# MAR (Li et al., 2024) as the paper ran it, in $DATA_ROOT/mar/:
#   mar/             the authors' repository (LTH14/mar) at commit c6d53f7, with the released MAR-B and KL-16
#                    VAE fetched by its util/download.py (about 1.9 GB) and its FID statistics of the ADM
#                    ImageNet-256 reference batch
#   venv/            Python 3.11 with torch 2.5.1 (CUDA 12.4), numpy 2.4.6, Pillow 12.3.0 (the versions of the
#                    paper's runs), timm, and the torch-fidelity fork MAR's environment installs for FID against
#                    a statistics file; this repository is installed into it. venv_freeze.txt lists what it holds.
#   imagenet_train   a link to ImageNet-1k's train/ directory when IMAGENET_TRAIN is set: the kernel reads of
#                    Figure 4 (b) and Table 2 take 64 of its images (download them yourself under the
#                    ImageNet terms)
# The FID statistics are checked against the md5 of the paper's copy, the weights against the byte sizes of
# the paper's copies (no content hash of them was recorded, nor the commit of the torch-fidelity fork).
source "$(dirname "$0")/../scripts/_common.sh"
D="$DATA_ROOT/mar"
mkdir -p "$D"
[ -d "$D/mar/.git" ] || git clone https://github.com/LTH14/mar "$D/mar"
git -C "$D/mar" checkout --quiet c6d53f7fa6427634b5850ebed771b7c2d19ea21f
[ -x "$D/venv/bin/python" ] || "${PYTHON_FOR_VENV:-python3.11}" -m venv "$D/venv"
"$D/venv/bin/pip" install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
"$D/venv/bin/pip" install -q "numpy==2.4.6" "pillow==12.3.0" "timm==0.9.12" scipy tqdm requests \
  "git+https://github.com/LTH14/torch-fidelity.git@master#egg=torch-fidelity"
"$D/venv/bin/pip" install -q --no-deps -e "$REPO_ROOT"
"$D/venv/bin/pip" freeze > "$D/venv_freeze.txt"
(cd "$D/mar" && "$D/venv/bin/python" -c "from util.download import download_pretrained_vae as vae, \
download_pretrained_marb as marb; vae(); marb()")
"$D/venv/bin/python" - "$D/mar" <<'PY'
import hashlib, os, sys
os.chdir(sys.argv[1])
h = hashlib.md5(open("fid_stats/adm_in256_stats.npz", "rb").read()).hexdigest()
assert h == "dfc835a89b2fe4ca5c9a65ec3e9ecbd9", f"fid_stats/adm_in256_stats.npz md5 {h}"
for path, size in (("pretrained_models/vae/kl16.ckpt", 265900046),
                   ("pretrained_models/mar/mar_base/checkpoint-last.pth", 1663614946)):
    assert os.path.getsize(path) == size, f"{path}: {os.path.getsize(path)} bytes, expected {size}"
print("MAR ready: FID statistics match the paper's copy, weights match its sizes")
PY
if [ -n "${IMAGENET_TRAIN:-}" ]; then ln -sfn "$IMAGENET_TRAIN" "$D/imagenet_train"; fi
