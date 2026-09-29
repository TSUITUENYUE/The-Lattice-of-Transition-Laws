"""The text8 networks and one loader for all of them.

RotaryNetwork is a pre-norm transformer with rotary positions inside attention, attention going
through scaled_dot_product_attention: causal for the causal model, unmasked (reading mask tokens)
for the masked model and the lattice model. TwoStreamNetwork runs the same network over a content
stream and a query stream with attention causal in a given reveal order, so one forward pass
supervises every conditional along that order. LearnedPositionNetwork is a bidirectional
transformer on learned absolute positions; the graded and absorbing models of the channel pair and
the 26M lattice model use it. JointHead is the lattice model's co-trained head: a causal chain over
the advanced set that reads the network's features.

``SHIFT`` is how far a network's logits sit ahead of the coordinate they predict: a causal
network reads position i to write i+1, every other network writes the position it reads.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.text8 import MASK, V, VOC


def rope_tables(seq, head_dim, device, theta=10000.0):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device)
                           .float() / head_dim))
    ang = torch.arange(seq, device=device).float()[:, None] * inv[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1)
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1)
    return cos[None, None], sin[None, None]


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


class Block(nn.Module):
    """Pre-norm attention and MLP; dropout on both residual branches and on attention."""

    def __init__(self, d, heads, p_drop=0.0):
        super().__init__()
        self.h = heads
        self.hd = d // heads
        self.p = p_drop
        self.drop = nn.Dropout(p_drop)
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def forward(self, x, cos, sin, mask=None, causal=False, pos=None):
        """``mask`` (True = may attend) and ``pos`` (each token's rotary position) serve the
        two-stream network; the other networks attend by ``causal`` over positions 0..L-1."""
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q = q.view(B, L, self.h, self.hd).transpose(1, 2)
        k = k.view(B, L, self.h, self.hd).transpose(1, 2)
        v = v.view(B, L, self.h, self.hd).transpose(1, 2)
        c = cos if pos is None else cos[:, :, pos]
        s = sin if pos is None else sin[:, :, pos]
        q, k = apply_rope(q, c, s), apply_rope(k, c, s)
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=causal,
            dropout_p=(self.p if self.training else 0.0))
        x = x + self.drop(self.proj(o.transpose(1, 2).reshape(B, L, D)))
        return x + self.drop(self.mlp(self.n2(x)))


class RotaryNetwork(nn.Module):
    """Transformer on rotary positions; ``causal`` makes it the causal network."""

    def __init__(self, d=768, depth=12, heads=12, seq=256, causal=False, p_drop=0.0,
                 vocab=VOC, out_dim=V):
        super().__init__()
        self.d, self.heads, self.seq, self.causal = d, heads, seq, causal
        self.SHIFT = 1 if causal else 0
        self.tok = nn.Embedding(vocab, d)
        nn.init.normal_(self.tok.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(d, heads, p_drop)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, out_dim, bias=False)
        # angles for positions up to twice the window
        cos, sin = rope_tables(seq * 2, d // heads, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def features(self, x):
        L = x.shape[1]
        h = self.tok(x)
        for b in self.blocks:
            h = b(h, self.cos[:, :, :L], self.sin[:, :, :L], causal=self.causal)
        return self.norm(h)

    def forward(self, x):
        return self.out(self.features(x))


class TwoStreamNetwork(nn.Module):
    """The rotary network over two streams, attention causal in a reveal order.

    The content stream carries the tokens; the query stream carries a mask token per position
    with one learned query vector added. Both streams take the rotary angle of their position.
    Given rank[i], position i's index in the reveal order, content at i attends to content of
    rank at most rank[i], the query at i attends to content of rank below rank[i] and to itself,
    and the logits are read from the query stream: under the identity order this is next-token
    prediction, under a random order the any-order objective.
    """
    SHIFT = 0

    def __init__(self, d=768, depth=12, heads=12, seq=256, p_drop=0.0):
        super().__init__()
        self.seq = seq
        self.net = RotaryNetwork(d=d, depth=depth, heads=heads, seq=seq,
                                 causal=False, p_drop=p_drop)
        self.qvec = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.qvec, std=0.02)

    def build_mask(self, rank, dev):
        """rank[i] = position i's index in the reveal order; True = may attend."""
        L = len(rank)
        r = rank.to(dev)
        cc = r[None, :] <= r[:, None]
        qc = r[None, :] < r[:, None]
        allow = torch.zeros(2 * L, 2 * L, dtype=torch.bool, device=dev)
        allow[:L, :L] = cc
        allow[L:, :L] = qc
        allow[L:, L:] = torch.eye(L, dtype=torch.bool, device=dev)
        return allow

    def forward(self, x, rank):
        B, L = x.shape
        dev = x.device
        t = self.net
        content = t.tok(x)
        query = t.tok(torch.full_like(x, MASK)) + self.qvec
        h = torch.cat([content, query], dim=1)
        pos = torch.cat([torch.arange(L, device=dev),
                         torch.arange(L, device=dev)])
        allow = self.build_mask(rank, dev)
        for b in t.blocks:
            h = b(h, t.cos, t.sin, mask=allow, causal=False, pos=pos)
        return t.out(t.norm(h[:, L:]))


class LearnedPositionNetwork(nn.Module):
    """Bidirectional transformer over mask tokens on learned absolute positions."""

    SHIFT = 0

    def __init__(self, d=768, depth=12, heads=12, seq=256, p_drop=0.0,
                 vocab=VOC, out_dim=V):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        nn.init.normal_(self.tok.weight, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, seq, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d, heads, 4 * d, dropout=p_drop, batch_first=True,
            norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, out_dim)

    def features(self, x):
        h = self.tok(x) + self.pos[:, :x.shape[1]]
        return self.norm(self.enc(h))

    def forward(self, x):
        return self.out(self.features(x))


class JointHead(nn.Module):
    """Causal chain over one advanced set.

    Step i reads the network features at s_1..s_i, the teacher-forced tokens of the set before
    it, and the full feature sequence, through which the dependence between the set's positions
    is mediated.
    """

    def __init__(self, d_net=768, d=256, depth=2, heads=4, kmax=32, seq=256):
        super().__init__()
        self.proj = nn.Linear(d_net, d)
        self.prev = nn.Embedding(VOC, d)
        nn.init.normal_(self.prev.weight, std=0.02)
        self.pos_self = nn.Embedding(seq, d)
        self.pos_prev = nn.Embedding(seq + 1, d)
        nn.init.normal_(self.pos_self.weight, std=0.02)
        nn.init.normal_(self.pos_prev.weight, std=0.02)
        self.step = nn.Parameter(torch.zeros(1, kmax, d))
        nn.init.trunc_normal_(self.step, std=0.02)
        self.mem = nn.Linear(d_net, d)
        self.mem_pos = nn.Embedding(seq, d)
        nn.init.normal_(self.mem_pos.weight, std=0.02)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "n1": nn.LayerNorm(d),
                "xa": nn.MultiheadAttention(d, heads, batch_first=True),
                "n2": nn.LayerNorm(d),
                "sa": nn.MultiheadAttention(d, heads, batch_first=True),
                "n3": nn.LayerNorm(d),
                "ff": nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                    nn.Linear(4 * d, d)),
            }))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, V)

    def forward(self, feats, set_tokens, S, memory):
        """feats (B, k, d_net): features at the set's positions S (B, k), in position order;
        set_tokens (B, k): the true characters there; memory (B, L, d_net): all features."""
        B, k, _ = feats.shape
        dev = feats.device
        prev = torch.cat([torch.full((B, 1), MASK, device=dev,
                                     dtype=torch.long),
                          set_tokens[:, :-1]], dim=1)
        bos = self.pos_prev.num_embeddings - 1
        prev_pos = torch.cat([torch.full((B, 1), bos, device=dev,
                                         dtype=torch.long),
                              S[:, :-1]], dim=1)
        h = (self.proj(feats) + self.prev(prev) + self.pos_self(S)
             + self.pos_prev(prev_pos) + self.step[:, :k])
        mem = self.mem(memory) + self.mem_pos.weight.unsqueeze(0)
        cmask = torch.triu(torch.ones(k, k, device=dev, dtype=torch.bool),
                           diagonal=1)
        for L in self.layers:
            q = L["n1"](h)
            h = h + L["xa"](q, mem, mem, need_weights=False)[0]
            q = L["n2"](h)
            h = h + L["sa"](q, q, q, attn_mask=cmask,
                            need_weights=False)[0]
            h = h + L["ff"](L["n3"](h))
        return self.out(self.norm(h))


# the models, the networks each one trains (the first is its default) and the reading it takes:
# a causal network reads prefixes, a two-stream network reads through its order, every other
# network reads mask tokens
MODELS = ("causal", "masked", "two-stream", "lattice", "graded", "absorbing")
NETWORKS = {"causal": ("rotary",), "masked": ("rotary",), "two-stream": ("rotary",),
            "lattice": ("rotary", "learned"), "graded": ("learned",),
            "absorbing": ("learned",)}
READS_AS = {"causal": "causal", "masked": "masked", "two-stream": "two-stream",
            "lattice": "masked", "graded": "masked", "absorbing": "masked"}


def build(model, network=None, d=768, depth=12, heads=12, seq=256, p_drop=0.0, vocab=VOC):
    """The network a model of this type trains: the two-stream network for the two-stream
    model, the rotary network (causal for the causal model) or the learned-position network."""
    network = network or NETWORKS[model][0]
    if network not in NETWORKS[model]:
        raise ValueError(f"the {model} model trains the {' or '.join(NETWORKS[model])} "
                         f"network, not the {network} one")
    if model == "two-stream":
        return TwoStreamNetwork(d=d, depth=depth, heads=heads, seq=seq, p_drop=p_drop)
    if network == "rotary":
        return RotaryNetwork(d=d, depth=depth, heads=heads, seq=seq,
                             causal=(model == "causal"), p_drop=p_drop)
    return LearnedPositionNetwork(d=d, depth=depth, heads=heads, seq=seq, p_drop=p_drop,
                                  vocab=vocab)


def load(path, kind=None, device="cuda"):
    """A trained network in eval mode (its averaged weights), with the checkpoint's metadata.

    The checkpoint records the model type and the architecture, so the network is rebuilt from
    it. ``kind`` is the reading a caller intends (causal, masked or two-stream); a checkpoint
    trained as a model that reads otherwise is refused, since the weights of a causal and a
    masked network share their shapes and would load into the wrong reading. The metadata holds
    the step the weights are from (for a ``_best.pt`` file, the step it was banked at) and the
    validation readings that banked it.
    """
    ck = torch.load(path, map_location=device, weights_only=True)
    reads = READS_AS[ck["model"]]
    if kind is not None and kind != reads:
        raise SystemExit(
            f"{path} holds the {ck['model']} model, which reads as {reads!r}, and this "
            f"reading asked for {kind!r}.")
    net = build(ck["model"], network=ck["network"], d=ck["dim"], depth=ck["depth"],
                heads=ck["heads"], seq=ck["seq"], vocab=ck["vocab"]).to(device)
    net.load_state_dict(ck["ema"])
    net.eval()
    meta = {k: ck[k] for k in ("model", "network", "name", "step", "best_step", "score",
                               "readings") if k in ck}
    return net, meta
