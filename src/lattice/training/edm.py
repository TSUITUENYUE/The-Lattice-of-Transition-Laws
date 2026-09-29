"""EDM's training recipe for a CIFAR-10 model whose level varies across the image.

The models of a comparison differ in their training distribution (training/measures.py) and in
nothing else: the per-pixel preconditioning (models/pixel.py), the lambda weighting, Adam at
learning rate 1e-3 after a linear ramp over the first ten million images, the EMA with a
half-life of 500,000 images, and the resume behaviour are the recipe's and live here.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import torch

from ..data.cifar10 import SDATA


def edm_loss(model, x0, onehot, sigma_map):
    """Denoising regression under EDM's weighting, with the level given pixel by pixel."""
    yn = x0 + sigma_map * torch.randn_like(x0)
    lam = (sigma_map ** 2 + SDATA ** 2) / (sigma_map * SDATA) ** 2
    return (lam * (model.denoise(yn, sigma_map, onehot) - x0) ** 2).mean()


def _get_rng(generator):
    return {"generator": generator.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng(state, generator):
    # every random-number state is restored from a CPU tensor, whatever device it was loaded to
    # (set_state refuses a tensor on a card)
    generator.set_state(state["generator"].cpu())
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def run(model, ema, cifar, steps, out, tag, loss_fn, generator, lr=1e-3, bs=512, micro_bs=128,
        resume=True, log_every=500, save_every=5000, extra_state=None, roll_minutes=20.0,
        start_from=None):
    """Train one model to ``steps`` and save its EMA as OUT/ckpt_<tag>.pt.

    ``loss_fn(model, x0, onehot)`` is the objective on one micro-batch; the images, their flips and
    everything ``loss_fn`` draws on the CPU come from ``generator``. The weights, the optimiser
    state and every random-number state are saved as OUT/res_<tag>.pt every ``save_every`` steps
    and every ``roll_minutes`` of wall clock, through a temporary file and a rename. A restart
    resumes from that file, and ``start_from`` names another training's rolling state to begin
    from when there is none yet (a longer training of the same model); either way the run draws
    exactly the data an uninterrupted run would have drawn, and no step repeats an earlier one.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    roll = out / f"res_{tag}.pt"
    naccum = bs // micro_bs
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    ema_decay = 0.5 ** (bs / (500 * 1000))
    start = 0
    src = roll if roll.exists() else (Path(start_from) if start_from else None)
    if resume and src is not None:
        # loaded to the CPU: load_state_dict copies weights and optimiser state to the model's
        # device, and the random-number states must stay CPU tensors
        rs = torch.load(src, map_location="cpu", weights_only=False)
        model.load_state_dict(rs["model"])
        ema.load_state_dict(rs["ema"])
        opt.load_state_dict(rs["opt"])
        start = rs["step"]
        _set_rng(rs["rng"], generator)
        print(f"{tag}: resumed from {src} at step {start}", flush=True)
    t0 = time.time()
    last_roll = time.time()

    def save_roll(step_done):
        tmp = roll.with_suffix(".pt.tmp")
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "step": step_done, "tag": tag, "rng": _get_rng(generator),
                    **(extra_state or {})}, tmp)
        os.replace(tmp, roll)

    for it in range(start, steps):
        cur_lr = lr * min((it * bs) / 1e7, 1.0)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(naccum):
            x0, oh = cifar.batch(micro_bs, generator)
            loss = loss_fn(model, x0, oh) / naccum
            loss.backward()
            tot += float(loss.detach())
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(ema_decay).add_(pm, alpha=1 - ema_decay)
        if it % log_every == 0:
            print(f"LOSS it={it} loss={tot:.4f} lr={cur_lr:.2e} ({time.time() - t0:.0f}s)", flush=True)
        if (it + 1) % save_every == 0 or time.time() - last_roll >= 60.0 * roll_minutes:
            save_roll(it + 1)
            last_roll = time.time()
    torch.save({"ema": ema.state_dict(), "step": steps, "tag": tag, **(extra_state or {})},
               out / f"ckpt_{tag}.pt")
    print(f"TRAIN_DONE {tag}", flush=True)
    return ema
