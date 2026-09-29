"""Memory depth on the released SkyReels-V2 video model: rollouts at five depths, scored by VBench.

    python -m lattice.experiments.video_memory generate --skyreels DATA/video/SkyReels-V2 \
        --model DATA/video/SkyReels-V2-DF-1.3B-540P --prompt-index 0 --overlap 25 --out DIR
    python -m lattice.experiments.video_memory score --videos DIR --out DIR [--dimensions D,...]

generate: one rollout of 773 frames (eight times the 97-frame training window) of one of the
three prompts, seed 0, by the released Diffusion Forcing pipeline on its synchronised schedule
(ar_step 0, causal block 1): 30 UniPC steps per segment, guidance 6, flow shift 8, 544 x 960 at
24 fps, conditioning noise 20. The memory depth is the pipeline's overlap_history: every segment
after the first conditions on the last `overlap` frames of the video so far (5, 17 as shipped,
25, 33 or 57). All frame counts are 4n + 1, and every segment's step matrix is checked against the
plan (models/skyreels.py), so no conditioning frame is dropped and every frame of the video is
denoised in all 30 steps. The decoding steps of a rollout are 30 per segment. Writes
OUT/p<prompt>_ov<overlap>_s<seed>/video.mp4 and run.json (the settings, the checked segments, the
pixels' sha256, seconds and library versions); a finished rollout is not generated again.

score: VBench's subject consistency, background consistency, motion smoothness and imaging quality
(the dimensions declared before decoding; --dimensions names fewer) of every rollout under --videos,
each read over the whole rollout, so the seams between segments are inside the measurement. Writes
OUT/scores.csv (prompt, overlap, seed, dimension, score).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from importlib import metadata
from pathlib import Path

from ..models.skyreels import (ADDNOISE, AR_STEP, BASE_FRAMES, CAUSAL_BLOCK, FPS, GUIDANCE, HEIGHT,
                               NEGATIVE_PROMPT, PROMPTS, ROLLOUT_FRAMES, SHIFT, STEPS, WIDTH,
                               checked_pipeline, seeded_noise, segment_plan)
from ..utils.records import atomic_json

OVERLAPS = (5, 17, 25, 33, 57)


def generate(args):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    out = Path(args.out) / f"p{args.prompt_index}_ov{args.overlap}_s{args.seed}"
    if (out / "run.json").exists() and json.loads((out / "run.json").read_text())["status"] == "complete":
        print(f"{out.name}: finished earlier", flush=True)
        return
    plan = segment_plan(args.frames, BASE_FRAMES, args.overlap)
    versions = {}
    for name in ("torch", "diffusers", "transformers", "numpy", "imageio"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    record = dict(status="generating", prompt_index=args.prompt_index, prompt=PROMPTS[args.prompt_index],
                  overlap=args.overlap, frames=args.frames, seed=args.seed, base_frames=BASE_FRAMES,
                  steps_per_segment=STEPS, guidance=GUIDANCE, shift=SHIFT, addnoise=ADDNOISE, ar_step=AR_STEP,
                  causal_block=CAUSAL_BLOCK, height=HEIGHT, width=WIDTH, fps=FPS, plan=plan, versions=versions)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "run.json", record)
    pipe = checked_pipeline(args.skyreels, args.model, plan, device=args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with seeded_noise(generator), torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
        frames = pipe(prompt=PROMPTS[args.prompt_index], negative_prompt=NEGATIVE_PROMPT, height=HEIGHT,
                      width=WIDTH, num_frames=args.frames, num_inference_steps=STEPS, shift=SHIFT,
                      guidance_scale=GUIDANCE, generator=generator, overlap_history=args.overlap,
                      addnoise_condition=ADDNOISE, base_num_frames=BASE_FRAMES, ar_step=AR_STEP,
                      causal_block_size=CAUSAL_BLOCK, fps=FPS)[0]
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    if frames.shape != (args.frames, HEIGHT, WIDTH, 3) or len(pipe.segments) != len(plan):
        raise RuntimeError(f"got {frames.shape} frames in {len(pipe.segments)} segments, planned {len(plan)}")
    imageio.mimwrite(out / "video.mp4", frames, fps=FPS, quality=8, output_params=["-loglevel", "error"])
    record.update(status="complete", segments=pipe.segments, decoding_steps=sum(s["steps"] for s in pipe.segments),
                  pixel_sha256=hashlib.sha256(frames.tobytes()).hexdigest(), seconds=seconds,
                  peak_cuda_bytes=torch.cuda.max_memory_allocated())
    atomic_json(out / "run.json", record)
    print(json.dumps(dict(video=str(out / "video.mp4"), steps=record["decoding_steps"], seconds=round(seconds))),
          flush=True)


def score(args):
    from ..evaluation.video_quality import DIMENSIONS, score_videos
    videos, cells = {}, {}
    for run in sorted(Path(args.videos).glob("*/run.json")):
        record = json.loads(run.read_text())
        if record["status"] != "complete":
            raise SystemExit(f"{run.parent} is not finished")
        tag = run.parent.name
        videos[tag] = run.parent / "video.mp4"
        cells[tag] = dict(prompt=record["prompt_index"], overlap=record["overlap"], seed=record["seed"])
    dims = args.dimensions.split(",") if args.dimensions else list(DIMENSIONS)
    unknown = set(dims) - set(DIMENSIONS)
    if unknown:
        raise SystemExit(f"unknown dimensions {sorted(unknown)}; the declared ones are {', '.join(DIMENSIONS)}")
    rows = [dict(cells[tag], dimension=dim, score=value)
            for tag, dim, value in score_videos(videos, dims, Path(args.out) / "vbench", args.device)]
    with open(Path(args.out) / "scores.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["prompt", "overlap", "seed", "dimension", "score"])
        w.writeheader()
        w.writerows(rows)
    print(f"scored {len(videos)} rollouts on {', '.join(dims)}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["generate", "score"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--skyreels", help="the SkyReels-V2 checkout (generate)")
    ap.add_argument("--model", help="the SkyReels-V2-DF-1.3B-540P directory (generate)")
    ap.add_argument("--prompt-index", type=int, choices=range(len(PROMPTS)), default=0)
    ap.add_argument("--overlap", type=int, choices=OVERLAPS, default=17, help="memory depth, in frames")
    ap.add_argument("--frames", type=int, default=ROLLOUT_FRAMES, help="frames of the rollout (4n + 1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--videos", help="the directory of the rollouts (score)")
    ap.add_argument("--dimensions", default="", help="score: VBench dimensions, comma separated (default: the "
                    "four declared)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if args.stage == "generate":
        if not (args.skyreels and args.model):
            ap.error("generate needs --skyreels and --model")
        generate(args)
    else:
        if not args.videos:
            ap.error("score needs --videos")
        score(args)


if __name__ == "__main__":
    main()
