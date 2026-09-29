"""ImageNet-256 FID with ADM's evaluation suite.

guided-diffusion's evaluations/evaluator.py against the VIRTUAL_imagenet256_labeled.npz reference
batch, the evaluation SiT and DiT report (FID, sFID, Inception Score, precision, recall). It runs
in its own TensorFlow environment (setup/prepare_adm_evaluator.sh) and is held to the CPU, so the
numbers do not depend on the card.
"""
from __future__ import annotations

import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ADM_REF = "VIRTUAL_imagenet256_labeled.npz"


def pack_npz(sample_dir, out, n):
    """The evaluator reads one npz whose arr_0 is uint8 n x 256 x 256 x 3; written as a stream,
    so the samples never sit in memory at once."""
    sample_dir, out = Path(sample_dir), Path(out)
    names = [f"{i:06d}.png" for i in range(n)]
    missing = [x for x in names if not (sample_dir / x).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of {n} samples missing in {sample_dir}, first {missing[0]}")
    tmp = out.with_suffix(".part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        with zf.open("arr_0.npy", "w", force_zip64=True) as f:
            np.lib.format.write_array_header_1_0(
                f, {"descr": "|u1", "fortran_order": False, "shape": (n, 256, 256, 3)})
            for x in names:
                a = np.asarray(Image.open(sample_dir / x).convert("RGB"), dtype=np.uint8)
                if a.shape != (256, 256, 3):
                    raise SystemExit(f"{x} is {a.shape}")
                f.write(a.tobytes())
    tmp.replace(out)
    return out


def adm_score(sample_dir, n, adm_dir):
    """FID, sFID, Inception Score, precision and recall of the samples 000000.png .. (n-1).png.

    The samples are packed afresh for every score, beside the sample directory, and the packed
    file is deleted once the evaluator has read it."""
    sample_dir, adm = Path(sample_dir), Path(adm_dir)
    out = sample_dir.parent / f"{sample_dir.name}_adm{n}.npz"
    pack_npz(sample_dir, out, n)
    try:
        r = subprocess.run([str(adm / "venv/bin/python"), str(adm / "evaluator.py"), str(adm / ADM_REF), str(out)],
                           cwd=str(adm), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    finally:
        out.unlink(missing_ok=True)
    st = {}
    for line in r.stdout.splitlines():
        for key in ("Inception Score", "FID", "sFID", "Precision", "Recall"):
            if line.startswith(key + ":"):
                st[key] = float(line.split(":", 1)[1])
    if r.returncode != 0 or len(st) != 5:
        print(r.stdout[-3000:], flush=True)
        raise SystemExit(f"the ADM evaluator exited {r.returncode} and printed {sorted(st)}")
    return st
