"""The released SkyReels-V2-DF-1.3B-540P video model through its own Diffusion Forcing pipeline.

The pipeline comes from a checkout of SkyReels-V2 at a pinned commit (setup/prepare_video.sh). This
module holds the settings the paper uses and two additions made from outside the pipeline's code:

    checked_pipeline   a pipeline that checks every segment's step matrix against the segment
                       plan before denoising it: the synchronised schedule (ar_step 0) with causal
                       block 1, so no conditioning frame is dropped to fill a block, and every new
                       latent frame of the segment is updated in each of the 30 steps, so no frame
                       of the video is decoded from undenoised noise
    seeded_noise       during a pipeline call, the noise added to each segment's conditioning
                       frames is drawn from the call's generator instead of torch's global one, so
                       a video is reproduced exactly from its seed
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import torch

PROMPTS = (
    "A chef chops vegetables in a busy restaurant kitchen.",
    "A dog runs across a grassy field toward a frisbee.",
    "Waves break on a rocky shore at sunset.",
)
# the negative prompt of the released generate_video_df.py, unchanged
NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
HEIGHT, WIDTH, FPS = 544, 960, 24
BASE_FRAMES = 97          # the training window, in frames
ROLLOUT_FRAMES = 773      # a rollout of the paper, about eight times the training window
STEPS = 30                # UniPC steps per segment
GUIDANCE, SHIFT = 6.0, 8.0
ADDNOISE = 20             # noise level of the conditioning frames
AR_STEP, CAUSAL_BLOCK = 0, 1


def latent_frames(frames):
    return (frames - 1) // 4 + 1


def segment_plan(frames, base, overlap):
    """The segments of a rollout, in latent frames: each after the first conditions on the latent
    frames of the last `overlap` video frames and adds new ones. Every length must be 4n + 1 frames,
    so that no segment holds a partial latent frame."""
    if any(v < 1 or (v - 1) % 4 for v in (frames, base, overlap)):
        raise ValueError("frames, base and overlap must each be 4n + 1")
    if not 1 <= overlap < base < frames:
        raise ValueError("need 1 <= overlap < base < frames")
    total, window, prefix = latent_frames(frames), latent_frames(base), latent_frames(overlap)
    plan = [dict(index=0, latent_length=window, prefix_latents=0, prefix_start_frame=None,
                 prefix_end_frame=None, new_latents=window)]
    finished = window
    while finished < total:
        new = min(window - prefix, total - finished)
        old_frames = (finished - 1) * 4 + 1
        plan.append(dict(index=len(plan), latent_length=prefix + new, prefix_latents=prefix,
                         prefix_start_frame=old_frames - overlap, prefix_end_frame=old_frames - 1,
                         new_latents=new))
        finished += new
    return plan


def check_segment(expected, args, result):
    """The segment record, after checking one generate_timestep_matrix call against its plan."""
    length, _, _, ar_step, prefix, block = args[:6]
    if (ar_step, block) != (AR_STEP, CAUSAL_BLOCK):
        raise RuntimeError(f"segment {expected['index']}: ar_step {ar_step}, causal block {block}")
    if length != expected["latent_length"] or prefix != expected["prefix_latents"]:
        raise RuntimeError(f"segment {expected['index']}: {length} latent frames with {prefix} conditioning, "
                           f"planned {expected['latent_length']} with {expected['prefix_latents']}")
    matrix, _, mask, intervals = result
    updates = mask.sum(0).tolist()
    if tuple(matrix.shape) != (STEPS, length) or updates != [0] * prefix + [STEPS] * (length - prefix) \
            or any(tuple(iv) != (0, length) for iv in intervals):
        raise RuntimeError(f"segment {expected['index']}: a new latent frame is not updated in all {STEPS} steps")
    return dict(expected, steps=int(matrix.shape[0]), updates=updates)


def checked_pipeline(skyreels, model, plan, device="cuda"):
    """The released pipeline, loaded with its CPU offloading, checking each segment against plan;
    the checked segments collect in pipe.segments."""
    sys.path.insert(0, str(Path(skyreels).resolve()))
    from skyreels_v2_infer import DiffusionForcingPipeline

    class CheckedPipeline(DiffusionForcingPipeline):
        def generate_timestep_matrix(self, *args, **kwargs):
            result = super().generate_timestep_matrix(*args, **kwargs)
            self.segments.append(check_segment(plan[len(self.segments)], args, result))
            return result

    pipe = CheckedPipeline(str(model), dit_path=str(model), device=torch.device(device),
                           weight_dtype=torch.bfloat16, use_usp=False, offload=True)
    pipe.segments = []
    return pipe


@contextlib.contextmanager
def seeded_noise(generator):
    """Inside the block, torch.randn_like draws from `generator` (the pipeline adds the
    conditioning noise with torch.randn_like, and nothing else in a pipeline call uses it)."""
    from diffusers.utils.torch_utils import randn_tensor
    original = torch.randn_like
    torch.randn_like = lambda t, *a, **k: randn_tensor(t.shape, generator=generator, device=t.device, dtype=t.dtype)
    try:
        yield
    finally:
        torch.randn_like = original
