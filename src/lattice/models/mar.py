"""MAR-B and its KL-16 tokenizer, from a checkout of the authors' repository (setup/prepare_mar.sh).

The checkout's own model code is imported, and its released MAR-B (EMA weights) and KL-16 VAE are
loaded from its pretrained_models/. Nothing is downloaded here.
"""
import os
import sys

import torch

CHECKPOINT = os.path.join("pretrained_models", "mar", "mar_base", "checkpoint-last.pth")
VAE = os.path.join("pretrained_models", "vae", "kl16.ckpt")
FID_STATS = os.path.join("fid_stats", "adm_in256_stats.npz")


def load(mar_dir, device, num_sampling_steps="100"):
    """(model, vae): MAR-B with its EMA weights and the KL-16 VAE, in eval mode on the device,
    built as the authors' main_mar.py builds them. Building draws the initial weights from torch's
    global generator before the released ones replace them, as the authors' code does."""
    mar_dir = os.path.abspath(mar_dir)
    if mar_dir not in sys.path:
        sys.path.insert(0, mar_dir)
    from models import mar
    from models.vae import AutoencoderKL
    model = mar.mar_base(img_size=256, vae_stride=16, patch_size=1, vae_embed_dim=16, mask_ratio_min=0.7,
                         label_drop_prob=0.1, class_num=1000, attn_dropout=0.1, proj_dropout=0.1,
                         buffer_size=64, diffloss_d=6, diffloss_w=1024, num_sampling_steps=num_sampling_steps,
                         diffusion_batch_mul=4, grad_checkpointing=False).to(device)
    model.load_state_dict(torch.load(os.path.join(mar_dir, CHECKPOINT), map_location="cpu")["model_ema"])
    model.eval()
    vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4),
                        ckpt_path=os.path.join(mar_dir, VAE)).to(device).eval()
    return model, vae
