"""VBench scores of generated videos: the four dimensions declared before decoding (subject
consistency, background consistency, motion smoothness, imaging quality), one per video.

Each dimension is VBench's own evaluation in its custom-input mode, over every frame of the video.
The videos are staged under their names in one directory (symbolic links), as that mode reads them.
Background consistency encodes the frames with CLIP in float32 and in batches of 32, since the
half-precision encoder's score changes with the encoding batch; VBench's reduction is unchanged.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

DIMENSIONS = ("subject_consistency", "background_consistency", "motion_smoothness", "imaging_quality")
CLIP_BATCH = 32                      # background consistency: frames per CLIP encoding call


def score_videos(videos, dimensions, out, device="cuda"):
    """[(name, dimension, score)] for the videos {name: path}, VBench's files kept under out."""
    import torch
    import vbench
    from vbench import VBench
    out = Path(out)
    stage = out / "videos"
    stage.mkdir(parents=True, exist_ok=True)
    for name, path in videos.items():
        link = stage / f"{name}.mp4"
        if not link.exists():
            link.symlink_to(Path(path).resolve())
    full_info = Path(vbench.__file__).parent / "VBench_full_info.json"
    scores = []
    for dim in dimensions:
        restore = _float32_clip(torch) if dim == "background_consistency" else None
        try:
            bench = VBench(torch.device(device), str(full_info), str(out / dim))
            with torch.no_grad():
                bench.evaluate(videos_path=str(stage), name=dim, dimension_list=[dim], mode="custom_input")
        finally:
            if restore is not None:
                restore()
        results = json.loads((out / dim / f"{dim}_eval_results.json").read_text())
        for item in results[dim][1]:
            scores.append((Path(item["video_path"]).stem, dim, float(item["video_results"])))
        del bench
        gc.collect()
        torch.cuda.empty_cache()
    missing = {(n, d) for n in videos for d in dimensions} - {(n, d) for n, d, _ in scores}
    if missing:
        raise RuntimeError(f"VBench returned no score for {sorted(missing)}")
    return scores


def _float32_clip(torch):
    """Make clip.load return CLIP in float32, encoding images CLIP_BATCH at a time; returns the undo."""
    import clip
    original = clip.load

    def load(*args, **kwargs):
        model, preprocess = original(*args, **kwargs)
        model.float()
        encode = model.encode_image

        def encode_in_batches(images):
            with torch.no_grad():
                return torch.cat([encode(batch) for batch in images.split(CLIP_BATCH)], dim=0)

        model.encode_image = encode_in_batches
        return model, preprocess

    clip.load = load

    def restore():
        clip.load = original
    return restore
