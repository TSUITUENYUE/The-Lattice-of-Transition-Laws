"""A level on every patch of an ImageNet-256 latent, on an unchanged SiT.

A SiT conditions on one level per sample, which is the synchronised corner of the lattice and the
only state a diffusion model is shown. Giving the level a patch axis puts the interior of the
lattice on the training support, so a schedule that advances part of an image queries a state the
weights were trained on.

Nothing in the network changes. A SiT block modulates its tokens with a vector per sample; the
same modules accept a vector per token, since the adaLN projection is pointwise in the last
dimension and the modulation is a broadcast multiply. A conditioning that is constant along the
token axis reproduces the block's own forward exactly (``check_constant_path``).
"""
from __future__ import annotations

import torch


def block_seqcond(block, x, c):
    """A SiT block whose conditioning carries a token axis."""
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
        block.adaLN_modulation(c).chunk(6, dim=-1)
    h = block.norm1(x) * (1 + scale_msa) + shift_msa
    x = x + gate_msa * block.attn(h)
    h = block.norm2(x) * (1 + scale_mlp) + shift_mlp
    return x + gate_mlp * block.mlp(h)


def final_seqcond(layer, x, c):
    shift, scale = layer.adaLN_modulation(c).chunk(2, dim=-1)
    return layer.linear(layer.norm_final(x) * (1 + scale) + shift)


def forward_perpatch(model, x, t_p, y):
    """SiT.forward with a level on every patch.

    ``t_p`` is (N, T) in [0, 1], one level per patch in the order the patch embedder produces
    (raster order over the token grid).
    """
    n = x.shape[0]
    x = model.x_embedder(x) + model.pos_embed
    temb = model.t_embedder(t_p.reshape(-1))
    temb = temb.view(n, t_p.shape[1], -1)
    yv = model.y_embedder(y, model.training).unsqueeze(1)
    c = temb + yv
    for block in model.blocks:
        x = block_seqcond(block, x, c)
    x = final_seqcond(model.final_layer, x, c)
    x = model.unpatchify(x)
    if model.learn_sigma:
        # the network carries a variance head this objective does not use, and drops it here;
        # the per-patch path drops the same half
        x, _ = x.chunk(2, dim=1)
    return x


def levels_to_pixels(t_p, side, patch):
    """The per-patch level (N, side * side), in raster order over the token grid, laid back over
    the latent pixels each patch covers: (N, 1, side * patch, side * patch)."""
    te = t_p.reshape(t_p.shape[0], 1, side, side)
    return te.repeat_interleave(patch, -2).repeat_interleave(patch, -1)


def check_constant_path(model, dev, n=2, tokens=256, seed=0):
    """A level constant along the token axis must reproduce the network's own forward, and a
    level that varies must not.

    SiT zero-initialises every adaLN projection and the final linear, so a freshly built model
    outputs zeros and any agreement between two paths through it says nothing. On a model whose
    modulation is inert the check therefore runs on a copy with the projections and the output
    head switched on (weights drawn from a CPU generator), and says so by returning
    ``synthetic=True``; on trained weights it runs on the weights themselves. The copy is in eval
    mode, so SiT's label dropout cannot give the two paths different labels.

    Returns (agreement, movement, synthetic), the first two relative to the output scale.
    """
    import copy
    g = torch.Generator(device="cpu").manual_seed(seed)
    inert = sum(float(b.adaLN_modulation[-1].weight.detach().abs().max())
                for b in model.blocks) == 0.0
    m = copy.deepcopy(model).eval()
    if inert:
        with torch.no_grad():
            qs = [q for blk in m.blocks
                  for q in (blk.adaLN_modulation[-1].weight,
                            blk.adaLN_modulation[-1].bias)]
            qs += [m.final_layer.adaLN_modulation[-1].weight,
                   m.final_layer.adaLN_modulation[-1].bias,
                   m.final_layer.linear.weight, m.final_layer.linear.bias]
            for q in qs:
                q.copy_(torch.randn(q.shape, generator=g) * 0.05)
    x = torch.randn(n, 4, 32, 32, generator=g).to(dev)
    y = torch.randint(0, 1000, (n,), generator=g).to(dev)
    t = (torch.rand(n, generator=g) * 0.8 + 0.1).to(dev)
    tp = (torch.rand(n, tokens, generator=g) * 0.8 + 0.1).to(dev)
    # full fp32 for the three forwards, since a card whose default runs fp32 matmuls in TF32
    # would part the two paths by about 1e-3
    mm, cd = (torch.backends.cuda.matmul.allow_tf32,
              torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.no_grad():
            a = m(x, t, y)
            b = forward_perpatch(m, x, t[:, None].expand(-1, tokens), y)
            c = forward_perpatch(m, x, tp, y)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = mm
        torch.backends.cudnn.allow_tf32 = cd
    scale = max(float(a.abs().max()), 1e-12)
    return (float((a - b).abs().max()) / scale,
            float((c - b).abs().max()) / scale, inert)
