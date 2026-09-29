"""The kernel between frames of the released SkyReels-V2 video model, read on its own sampling states.

    python -m lattice.experiments.video_kernel --skyreels DATA/video/SkyReels-V2 \
        --model DATA/video/SkyReels-V2-DF-1.3B-540P --out DIR

For each of the three prompts (seeds 1234, 1235, 1236) one 97-frame window (25 latent frames) is
sampled from Gaussian noise on the synchronised schedule with the deployed settings (30 UniPC steps,
flow shift 8, guidance 6, the 24 fps embedding), each latent frame with its own solver. At steps 6,
15 and 24 of that trajectory, each of the latent frames 6, 12 and 18 in turn takes its next step (a
copy of its solver, with the solver's history, so the trajectory itself is not changed) and its
level advances with it; the guided prediction is read again with every other frame's state and level
unchanged. For every other frame the change is recorded two ways: the mean squared change of the
flow prediction over the mean square of the whole prediction, and the mean squared change of the
clean-frame estimate x - sigma v over that frame's mean squared estimate. Writes OUT/kernel.csv, one
row per prompt, step, moved frame and other frame (648 rows), and OUT/run.json; Table 28's display
reduces it to the kernel by distance that Appendix D quotes.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from pathlib import Path

from ..models.skyreels import FPS, GUIDANCE, HEIGHT, NEGATIVE_PROMPT, PROMPTS, SHIFT, STEPS, WIDTH
from ..utils.records import atomic_json

PROBE_STEPS = (6, 15, 24)
SOURCES = (6, 12, 18)
LATENT_FRAMES = 25               # a 97-frame window


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skyreels", required=True, help="the SkyReels-V2 checkout")
    ap.add_argument("--model", required=True, help="the SkyReels-V2-DF-1.3B-540P directory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(args.skyreels).resolve()))
    import numpy as np
    import torch
    from diffusers.utils.torch_utils import randn_tensor
    from skyreels_v2_infer import DiffusionForcingPipeline
    from skyreels_v2_infer.scheduler.fm_solvers_unipc import FlowUniPCMultistepScheduler

    record = dict(status="running", prompts=list(PROMPTS), seeds=[1234 + i for i in range(len(PROMPTS))],
                  latent_frames=LATENT_FRAMES, steps=STEPS, probe_steps=PROBE_STEPS, sources=SOURCES,
                  guidance=GUIDANCE, shift=SHIFT, fps=FPS)
    atomic_json(out / "run.json", record)
    started = time.perf_counter()
    pipe = DiffusionForcingPipeline(str(args.model), dit_path=str(args.model), device=torch.device("cuda"),
                                    weight_dtype=torch.bfloat16, use_usp=False, offload=True)
    tr = pipe.transformer
    pipe.text_encoder.to(pipe.device)
    with torch.no_grad():
        contexts = [pipe.text_encoder.encode(text).to(tr.dtype) for text in PROMPTS]
        negative = pipe.text_encoder.encode(NEGATIVE_PROMPT).to(tr.dtype)
    pipe.text_encoder.cpu()
    torch.cuda.empty_cache()
    tr.to(pipe.device)
    fps_embedding = [1]          # the pipeline embeds 16 fps as 0 and every other rate, 24 here, as 1

    def scheduler():
        value = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        value.set_timesteps(STEPS, device=pipe.device, shift=SHIFT)
        return value

    template = scheduler()
    ts, sigmas = template.timesteps, template.sigmas
    if not all(float(sigmas[k]) > float(sigmas[k + 1]) for k in range(STEPS)):
        raise RuntimeError("the noise levels of the schedule do not decrease")
    record["timesteps"], record["sigmas"] = ts.cpu().tolist(), sigmas.tolist()
    rows = []
    for pi, context in enumerate(contexts):
        seed = 1234 + pi
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        latents = randn_tensor((16, LATENT_FRAMES, HEIGHT // 8, WIDTH // 8), generator=generator,
                               device=pipe.device, dtype=tr.dtype)
        schedulers = [scheduler() for _ in range(LATENT_FRAMES)]

        def predict(x, times):
            with torch.no_grad(), torch.autocast("cuda", dtype=tr.dtype):
                conditional = tr(torch.stack([x]), t=times, context=context, fps=fps_embedding)[0]
                unconditional = tr(torch.stack([x]), t=times, context=negative, fps=fps_embedding)[0]
                return unconditional + GUIDANCE * (conditional - unconditional)

        for k, timestep in enumerate(ts):
            times = timestep.expand(LATENT_FRAMES).unsqueeze(0).clone()
            velocity = predict(latents, times)
            if not bool(torch.isfinite(velocity).all()):
                raise RuntimeError(f"non-finite prediction at step {k}")
            if k in PROBE_STEPS:
                sigma, sigma_next = float(sigmas[k]), float(sigmas[k + 1])
                flow_scale = velocity.float().square().mean().item()
                clean = latents.float() - sigma * velocity.float()
                for source in SOURCES:
                    probe = copy.deepcopy(schedulers[source])
                    probe_generator = torch.Generator(device="cuda")
                    probe_generator.set_state(generator.get_state())
                    moved = latents.clone()
                    moved[:, source] = probe.step(velocity[:, source], timestep, latents[:, source],
                                                  return_dict=False, generator=probe_generator)[0]
                    if schedulers[source].step_index != k or probe.step_index != k + 1:
                        raise RuntimeError("the probe's step moved the trajectory's own solver")
                    next_times = times.clone()
                    next_times[0, source] = ts[k + 1] if k + 1 < len(ts) else 0
                    after = predict(moved, next_times)
                    for target in range(LATENT_FRAMES):
                        if target == source:
                            continue
                        delta = after[:, target].float() - velocity[:, target].float()
                        delta_flow = delta.square().mean().item()
                        # the other frames keep their state and level, so the clean estimate moves by -sigma dv
                        delta_clean = (-sigma * delta).square().mean().item()
                        clean_scale = clean[:, target].square().mean().item()
                        rows.append(dict(prompt=pi, seed=seed, step=k, timestep=float(timestep), sigma=sigma,
                                         next_timestep=float(next_times[0, source]), next_sigma=sigma_next,
                                         source=source, target=target, distance=abs(target - source),
                                         flow_displacement_mse=delta_flow, flow_scale_global=flow_scale,
                                         relative_flow_mse=delta_flow / (flow_scale + 1e-12),
                                         clean_displacement_mse=delta_clean, clean_scale_target=clean_scale,
                                         relative_clean_mse=delta_clean / (clean_scale + 1e-12)))
                    print(f"prompt {pi} step {k} source {source}: {len(rows)} rows", flush=True)
            for frame, frame_scheduler in enumerate(schedulers):
                latents[:, frame] = frame_scheduler.step(velocity[:, frame], timestep, latents[:, frame],
                                                         return_dict=False, generator=generator)[0]
            if not bool(torch.isfinite(latents).all()):
                raise RuntimeError(f"non-finite state after step {k}")
    expected = len(PROMPTS) * len(PROBE_STEPS) * len(SOURCES) * (LATENT_FRAMES - 1)
    if len(rows) != expected:
        raise RuntimeError(f"{len(rows)} rows, expected {expected}")
    with open(out / "kernel.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    record.update(status="complete", rows=len(rows), seconds=time.perf_counter() - started)
    atomic_json(out / "run.json", record)
    print(json.dumps(dict(rows=len(rows), seconds=round(record["seconds"]))), flush=True)


if __name__ == "__main__":
    main()
