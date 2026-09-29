#!/usr/bin/env bash
# SkyReels-V2 as the paper ran it, in $DATA_ROOT/video/:
#   SkyReels-V2/                the authors' repository (SkyworkAI/SkyReels-V2) at commit 9351d131
#   SkyReels-V2-DF-1.3B-540P/   the released Diffusion Forcing weights, Hugging Face revision 11001117
#   venv/                       Python 3.11 with the libraries of the paper's runs (torch 2.5.1, diffusers 0.39.0,
#                               transformers 4.33.2, VBench 0.1.5, numpy 1.26.4, imageio 2.37.4) and the rest of
#                               SkyReels-V2's requirements, flash-attention included; this repository is installed
#                               into it. venv_freeze.txt lists what it holds.
# VBench fetches its own scoring models (DINO for subject consistency, MUSIQ for imaging quality) on first use.
source "$(dirname "$0")/../scripts/_common.sh"
D="$DATA_ROOT/video"
mkdir -p "$D"
[ -d "$D/SkyReels-V2/.git" ] || git clone https://github.com/SkyworkAI/SkyReels-V2 "$D/SkyReels-V2"
git -C "$D/SkyReels-V2" checkout --quiet 9351d13152207cc04de780e055346b08ade0b851
[ -x "$D/venv/bin/python" ] || "${PYTHON_FOR_VENV:-python3.11}" -m venv "$D/venv"
"$D/venv/bin/pip" install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
"$D/venv/bin/pip" install -q "diffusers==0.39.0" "transformers==4.33.2" "vbench==0.1.5" "numpy==1.26.4" \
  "imageio==2.37.4" imageio-ffmpeg "opencv-python==4.10.0.84" "accelerate==1.6.0" easydict ftfy decord \
  safetensors huggingface_hub tqdm
"$D/venv/bin/pip" install -q flash-attn --no-build-isolation
"$D/venv/bin/pip" install -q --no-deps -e "$REPO_ROOT"
"$D/venv/bin/pip" freeze > "$D/venv_freeze.txt"
"$D/venv/bin/python" - "$D" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("Skywork/SkyReels-V2-DF-1.3B-540P", revision="1100111771ba2d921f10e76991f54db9f73edb3d",
                  local_dir=f"{sys.argv[1]}/SkyReels-V2-DF-1.3B-540P")
print("SkyReels-V2 ready")
PY
