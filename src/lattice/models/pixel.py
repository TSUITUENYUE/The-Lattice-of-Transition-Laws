"""A CIFAR-10 model whose noise level varies across the image.

A diffusion model carries one level for the whole image, which is the synchronised path and the
only one it can express. Giving the level a spatial extent puts both corners of the lattice in
reach of one weight set: patch-sequential decoding is a level map that resolves one patch at a
time, synchronised denoising is a level map that is constant, and a sloped path carries one slope
across the patches.

The change to the network is the smallest one that carries the map. EDM's preconditioning is
already a function of the level, so it applies per pixel with no new algebra. The network (EDM's
SongUNet, class-conditional) is told the map through one extra input channel, ln(sigma) / 4 at
every pixel, and keeps its scalar level embedding, taken at the mean of that channel, which is
the value it would have had on the synchronised path.

The image is cut into square patches that share a level; a state of the lattice is a level index
per patch, 0 at the noisy end of the levels and L at the clean end.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data.cifar10 import SDATA
from . import edm_unet as EN

LOG_SCALE = 4.0


class LevelMapEDM(nn.Module):
    """D(y; sigma_map): EDM's SongUNet under EDM's preconditioning, applied per pixel."""

    def __init__(self, n_classes=10, channels=128, resolution=32, channel_mult=(2, 2, 2)):
        super().__init__()
        self.net = EN.SongUNet(img_resolution=resolution, in_channels=4, out_channels=3,
                               label_dim=n_classes, augment_dim=0, model_channels=channels,
                               channel_mult=list(channel_mult))

    def denoise(self, y, sigma_map, labels_onehot):
        s2 = sigma_map ** 2
        c_skip = SDATA ** 2 / (s2 + SDATA ** 2)
        c_out = sigma_map * SDATA / (s2 + SDATA ** 2).sqrt()
        c_in = 1.0 / (s2 + SDATA ** 2).sqrt()
        level = sigma_map.log() / LOG_SCALE
        c_noise = level.flatten(1).mean(1)
        x = torch.cat([c_in * y, level], dim=1)
        F = self.net(x, c_noise, labels_onehot, None)
        return c_skip * y + c_out * F

    forward = denoise


def patch_index(resolution=32, patch=8):
    """Which patch each pixel belongs to, patches numbered in raster order; and their count."""
    n = resolution // patch
    row = torch.arange(resolution) // patch
    idx = row.view(-1, 1) * n + row.view(1, -1)
    return idx, n * n


def level_map(levels, patches, knots):
    """The sigma map, (N, 1, H, W), of per-patch level indices (N, patches); ``knots`` holds sigma
    at every level index."""
    sig = knots[levels.clamp(0, len(knots) - 1)]
    return sig[:, patches].unsqueeze(1)


def load(path, device="cuda"):
    """A trained model (its EMA weights, in eval mode) and the checkpoint's recorded configuration."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model = LevelMapEDM(channels=ck["channels"]).to(device)
    model.load_state_dict(ck["ema"])
    model.eval()
    return model, ck
