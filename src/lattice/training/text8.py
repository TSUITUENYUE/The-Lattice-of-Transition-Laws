"""Training the text8 models: each model's objective, its validation reading, and one loop.

What one micro-batch trains:
  causal       next-character cross-entropy, on the rotary network with causal attention.
  masked       the masked-diffusion objective: a rate u per row, drawn uniformly on [0.01, 1];
               each position masked with probability u, plus one position per row drawn
               uniformly, so every row scores something; every masked position scored with
               weight 1/u.
  two-stream   every conditional along one reveal order: the identity order with probability
               0.5 and a uniform permutation otherwise, so half of the micro-batches train
               exactly the left-to-right order the AR-chain reading follows.
  lattice      one network and its co-trained joint head on three supports in equal shares:
               prefix states (a window revealed up to a position drawn uniformly from 1..L-1 and
               masked from there, that position scored), random-subset masking (the masked
               objective) and advanced sets (a width k drawn from 2, 4, 8, 16, 32 per
               micro-batch; each window masked at a rate drawn uniformly on [0.05, 0.95] but at
               least (k + 2)/L + 0.02, plus the forced position, and topped up to k masked
               positions if short; k of its masked positions, drawn at random, are scored through
               the joint head as one causal chain in position order). The loss is the mean of
               the three. A third of the batch, rounded, goes to each support and is split over
               the micro-batches: a batch of 512 in micro-batches of 256 trains 85 windows of
               each support per micro-batch, 510 per step.
  graded,      the channel pair, on the learned-position network: every coordinate is set to a
  absorbing    level drawn uniformly and independently (the mask, the class or the character for
               the graded model; the mask or the character for the absorbing model), and every
               coordinate below the top level is scored, weighted by the reciprocal of the
               unresolved fraction of its row. The absorbing model therefore sees each coordinate
               masked independently with probability one half and never a heavily masked state;
               its objective is not the masked-diffusion objective, whose rate is drawn per row.

The loop is shared. AdamW with weight decay 0.01: the causal, masked, graded and absorbing models
decay only their weight matrices (norms, biases, embeddings and learned positions are not
decayed), the two-stream and lattice models decay every parameter. The learning rate warms up
linearly over 3,000 steps under a cosine that reaches zero at the last step; gradients are
clipped at norm 1; an averaged copy of the weights (decay 0.999) is what every reader loads.

Checkpoint selection. Every ``val_every`` steps the averaged weights are read in eval mode
(dropout off) on validation text that stays fixed for the whole run, and banked as
``<name>_best.pt`` whenever the reading reaches a new minimum; the file records the step it was
banked at. The causal, masked, graded and absorbing models read 512 windows spread evenly over
valid.bin: next-character bpc; the masked-diffusion bound over four fixed draws of rates and
masks (the forced position included, as in training; the reading only ranks checkpoints of one
model); the channel's own objective over four fixed draws of levels. The two-stream model reads
256 validation windows under the identity order and one random order and is banked at its
identity-order bpc. The lattice model reads 512 windows at prefix states, the masked-diffusion
bound on four sets of 512 windows, and advanced sets of width 16 through the joint head and
through the network alone, and is banked at the mean of its prefix and masked readings. The
windows and draws of these two are drawn once from a fixed seed, never from the training
generator, so validation leaves the training draws untouched.

A rolling state (``<name>_state.pt``) is written at every validation and every 20 minutes; a
rerun resumes from it and restores every random-number state it saved, so a resumed run draws
what an uninterrupted one would have drawn. ``<name>_final.pt`` holds the averaged weights after
the last step, with the training time.
"""
from __future__ import annotations

import copy
import csv
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from ..data.text8 import ABSORBING, LN2, MASK, V, VOC, Graded, Text8, fixed_rates, \
    fixed_windows, mask_batch, windows
from ..models.text import NETWORKS, JointHead, build

VAL_SEED = 4242                 # the seed of every fixed validation draw
WIDTHS = (2, 4, 8, 16, 32)      # widths of the lattice model's advanced sets
P_IDENTITY = 0.5                # the two-stream model's probability of the identity order


def bf16(device):
    """bfloat16 autocast on a GPU; full precision on the CPU."""
    if torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


# --------------------------------------------------------------- objectives

def next_character_loss(net, x):
    """Next-character cross-entropy: the causal model."""
    with bf16(x.device):
        lg = net(x[:, :-1])
        return F.cross_entropy(lg.reshape(-1, V), x[:, 1:].reshape(-1))


def masked_loss(net, x, generator):
    """The masked-diffusion objective: the masked model, and the lattice model's random-subset
    support. The rate is drawn per row and the weight is its reciprocal, the continuous-time
    bound written as an expectation over rates."""
    B, L = x.shape
    u = torch.rand(B, generator=generator).clamp(0.01, 1.0)
    xin, m = mask_batch(x.cpu(), u, generator)
    xin, m = xin.to(x.device), m.to(x.device)
    with bf16(x.device):
        lg = net(xin)
        ce = F.cross_entropy(lg[m], x[m], reduction="none")
        w = (1.0 / u).to(x.device).unsqueeze(1).expand_as(m)[m]
        return (w * ce).sum() / (B * L)


def channel_loss(net, x, channel, generator):
    """The channel pair's objective: a level drawn uniformly for every coordinate, the state
    rendered by the channel, every coordinate below the top level scored with weight one over
    the unresolved fraction of its row. On the absorbing channel each coordinate is masked with
    probability one half."""
    B, L = x.shape
    lv = channel.draw_levels(x.shape, generator).to(x.device)
    obs = channel.render(x, lv)
    m = lv < channel.levels - 1
    frac = m.float().mean(1).clamp_min(1.0 / L)
    with bf16(x.device):
        lg = net(obs)
        ce = F.cross_entropy(lg[m], x[m], reduction="none")
        w = (1.0 / frac).unsqueeze(1).expand_as(m)[m]
        return (w * ce).sum() / (B * L)


def reveal_order(seq, generator):
    """The two-stream model's order for one micro-batch, as ranks (rank[i] is position i's
    place in the order): the identity with probability P_IDENTITY, else a uniform
    permutation."""
    if float(torch.rand(1, generator=generator)) < P_IDENTITY:
        return torch.arange(seq)
    return torch.argsort(torch.randperm(seq, generator=generator))


def two_stream_loss(net, x, rank):
    """Cross-entropy of every position's conditional along the order ``rank``."""
    with bf16(x.device):
        lg = net(x, rank)
        return F.cross_entropy(lg.reshape(-1, V), x.reshape(-1))


def prefix_batch(data, n, seq, generator):
    """Prefix states: n windows, each revealed before a position j drawn uniformly from
    1..seq-1 and masked from j on; j is the position scored."""
    x = windows(data, n, seq, generator)
    j = torch.randint(1, seq, (n,), generator=generator)
    xin = x.clone()
    xin[torch.arange(seq).unsqueeze(0) >= j.unsqueeze(1)] = MASK
    return xin, x, j


def width_batch(data, n, k, seq, generator):
    """Advanced sets of width k: n windows masked at a rate drawn uniformly on [0.05, 0.95] but
    at least (k + 2)/seq + 0.02 (with mask_batch's forced position), topped up with randomly
    chosen positions when fewer than k are masked; the set is k of the masked positions drawn at
    random, in position order."""
    x = windows(data, n, seq, generator)
    u = (0.05 + 0.90 * torch.rand(n, generator=generator)).clamp(min=(k + 2) / seq + 0.02)
    xin, m = mask_batch(x, u, generator)
    S = torch.zeros(n, k, dtype=torch.long)
    for i in range(n):
        pos = m[i].nonzero(as_tuple=False).view(-1)
        if len(pos) < k:
            rev = (~m[i]).nonzero(as_tuple=False).view(-1)
            add = rev[torch.randperm(len(rev), generator=generator)[:k - len(pos)]]
            m[i, add] = True
            xin[i, add] = MASK
            pos = m[i].nonzero(as_tuple=False).view(-1)
        perm = torch.randperm(len(pos), generator=generator)[:k]
        S[i] = pos[perm].sort().values
    return xin, x, S


def lattice_loss(net, head, train, sizes, seq, generator, device):
    """One micro-batch of the lattice model: the mean of its three supports' losses."""
    n_prefix, n_masked, n_width = sizes
    xin, x, j = prefix_batch(train, n_prefix, seq, generator)
    xin, x, j = xin.to(device), x.to(device), j.to(device)
    with bf16(device):
        lg = net(xin)
        rows = torch.arange(n_prefix, device=device)
        ce_prefix = F.cross_entropy(lg[rows, j].float(), x[rows, j])
    ce_masked = masked_loss(net, windows(train, n_masked, seq, generator).to(device),
                            generator)
    k = WIDTHS[int(torch.randint(0, len(WIDTHS), (1,), generator=generator))]
    xin, x, S = width_batch(train, n_width, k, seq, generator)
    xin, x, S = xin.to(device), x.to(device), S.to(device)
    with bf16(device):
        feats = net.features(xin)
        at_set = feats.gather(1, S.unsqueeze(-1).expand(-1, -1, feats.shape[-1]))
        tokens = x.gather(1, S)
        lg = head(at_set.float(), tokens, S, feats.float())
        ce_width = F.cross_entropy(lg.reshape(-1, V).float(), tokens.reshape(-1))
    return (ce_prefix + ce_masked + ce_width) / 3


def support_sizes(bs, micro_bs):
    """Windows of each lattice support per micro-batch: a third of the batch rounded for the
    prefix and random-subset supports, the rest for the advanced sets, each split evenly over
    the bs // micro_bs micro-batches (rounding down)."""
    n_prefix = max(1, round(bs / 3))
    n_masked = max(1, round(bs / 3))
    n_width = max(1, bs - n_prefix - n_masked)
    accum = max(1, bs // micro_bs)
    return accum, tuple(max(1, n // accum) for n in (n_prefix, n_masked, n_width))


# --------------------------------------------------------------- validation readings

@torch.no_grad()
def causal_reading(net, vwin, device):
    """Next-character bpc on fixed windows."""
    x = vwin.to(device)
    with bf16(device):
        lg = net(x[:, :-1])
    return float(F.cross_entropy(lg.float().reshape(-1, V), x[:, 1:].reshape(-1))) / LN2


@torch.no_grad()
def masked_reading(net, vwin, device, draws=4):
    """The masked-diffusion bound on fixed windows, over ``draws`` fixed draws of rates and
    masks."""
    x = vwin.to(device)
    tot = 0.0
    for d in range(draws):
        u = fixed_rates(vwin.shape[0], VAL_SEED + d)
        xin, m = mask_batch(vwin, u, torch.Generator().manual_seed(VAL_SEED + 7 * d + 1))
        xin, m = xin.to(device), m.to(device)
        with bf16(device):
            lg = net(xin)
        ce = F.cross_entropy(lg[m].float(), x[m], reduction="none")
        w = (1.0 / u).to(device).unsqueeze(1).expand_as(m)[m]
        tot += float((w * ce).sum() / (vwin.shape[0] * vwin.shape[1]))
    return tot / draws / LN2


@torch.no_grad()
def channel_reading(net, vwin, channel, device, draws=4):
    """The channel's own objective on fixed windows over ``draws`` fixed draws of levels; it
    compares checkpoints of one model and says nothing across models."""
    x = vwin.to(device)
    B, L = vwin.shape
    tot = 0.0
    for d in range(draws):
        g = torch.Generator().manual_seed(VAL_SEED + 13 * d)
        lv = channel.draw_levels(vwin.shape, g)
        obs = channel.render(vwin, lv).to(device)
        lv = lv.to(device)
        m = lv < channel.levels - 1
        with bf16(device):
            lg = net(obs)
        ce = F.cross_entropy(lg[m].float(), x[m], reduction="none")
        frac = m.float().mean(1).clamp_min(1.0 / L)
        w = (1.0 / frac).unsqueeze(1).expand_as(m)[m]
        tot += float((w * ce).sum() / (B * L))
    return tot / draws / LN2


def two_stream_validation(valid, n, seq):
    """The two-stream model's validation text: n windows and one random order, fixed."""
    g = torch.Generator().manual_seed(VAL_SEED)
    x = windows(valid, n, seq, g)
    return x, torch.argsort(torch.randperm(seq, generator=g))


@torch.no_grad()
def two_stream_reading(net, batch, device):
    """bpc of every position's conditional under the identity order and under the fixed random
    order."""
    x, order = batch
    xv = x.to(device)
    with bf16(device):
        li = net(xv, torch.arange(xv.shape[1]))
    identity = float(F.cross_entropy(li.reshape(-1, V).float(), xv.reshape(-1))) / LN2
    with bf16(device):
        lr = net(xv, order)
    random_order = float(F.cross_entropy(lr.reshape(-1, V).float(), xv.reshape(-1))) / LN2
    return {"identity_bpc": identity, "random_order_bpc": random_order}


def lattice_validation(valid, n, seq):
    """The lattice model's validation text, drawn once from a fixed seed: n prefix states, four
    masked sets of n windows, n advanced sets of width 16."""
    g = torch.Generator().manual_seed(VAL_SEED)
    prefix = prefix_batch(valid, n, seq, g)
    masked = []
    for _ in range(4):
        x = windows(valid, n, seq, g)
        u = torch.rand(n, generator=g).clamp(0.01, 1.0)
        masked.append((x, u, *mask_batch(x, u, g)))
    return prefix, masked, width_batch(valid, n, 16, seq, g)


@torch.no_grad()
def lattice_reading(net, head, batch, device):
    """The lattice model's readings: prefix states, the masked-diffusion bound, and width-16 sets
    through the joint head and through the network alone (the product of its conditionals)."""
    (xin, x, j), masked, (xw, x16, S) = batch
    n, seq = x.shape
    xin, x, j = xin.to(device), x.to(device), j.to(device)
    with bf16(device):
        lg = net(xin)
    rows = torch.arange(n, device=device)
    prefix = F.cross_entropy(lg[rows, j].float(), x[rows, j]) / LN2
    tot = 0.0
    for xm, u, xim, m in masked:
        xm, xim, m = xm.to(device), xim.to(device), m.to(device)
        with bf16(device):
            lg = net(xim)
        ce = F.cross_entropy(lg[m].float(), xm[m], reduction="none")
        w = (1.0 / u).to(device).unsqueeze(1).expand_as(m)[m]
        tot += float((w * ce).sum() / (n * seq))
    masked_bpc = tot / len(masked) / LN2
    xw, x16, S = xw.to(device), x16.to(device), S.to(device)
    with bf16(device):
        feats = net.features(xw)
        lg = net.out(feats)
    at_set = feats.gather(1, S.unsqueeze(-1).expand(-1, -1, feats.shape[-1]))
    tokens = x16.gather(1, S)
    product = F.cross_entropy(lg.gather(1, S.unsqueeze(-1).expand(-1, -1, V))
                              .reshape(-1, V).float(), tokens.reshape(-1)) / LN2
    with bf16(device):
        lj = head(at_set.float(), tokens, S, feats.float())
    joint = F.cross_entropy(lj.reshape(-1, V).float(), tokens.reshape(-1)) / LN2
    return {"prefix_bpc": float(prefix), "masked_bpc": masked_bpc,
            "width16_joint_bpc": float(joint), "width16_product_bpc": float(product)}


# --------------------------------------------------------------- the loop

def decay_groups(net, wd):
    """Decay the weight matrices; leave norms, biases, embeddings and positions alone."""
    decay, plain = [], []
    for n, p in net.named_parameters():
        flat = p.dim() < 2 or "tok" in n or "emb" in n or "pos" in n
        (plain if flat else decay).append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": plain, "weight_decay": 0.0}]


def lr_at(it, steps, base, warmup):
    """The learning rate at step it: base, times a linear warm-up over the first ``warmup``
    steps, times a cosine that falls from one at step 0 to zero at ``steps``."""
    warm = min((it + 1) / warmup, 1.0)
    return base * warm * 0.5 * (1 + math.cos(math.pi * it / steps))


def frozen(net):
    """``net`` as an averaged copy: eval mode, no gradients."""
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


@torch.no_grad()
def update_average(averaged, nets, decay):
    """Move every averaged parameter a fraction 1 - decay of the way to the trained one."""
    for avg, net in zip(averaged.values(), nets.values()):
        for pa, pn in zip(avg.parameters(), net.parameters()):
            pa.mul_(decay).add_(pn, alpha=1 - decay)


def rng_state(generators=()):
    """Every random-number state a loop draws from: the global CPU generator, the CUDA
    generators when present, and the loop's own generators."""
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "generators": [g.get_state() for g in generators]}


def restore_rng(state, generators=()):
    """Set every random-number state back; each is moved to the CPU first, since the setters
    take CPU byte tensors whatever device the state was loaded onto."""
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    for g, s in zip(generators, state["generators"], strict=True):
        g.set_state(s.cpu())


def save(obj, path):
    """torch.save through a temporary file and a rename, so a kill never leaves half a file."""
    tmp = Path(f"{path}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def append_row(path, row):
    """Append one validation reading to a CSV, writing the header with the first row."""
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def trim_log(path, step):
    """Drop the rows a resumed run will write again: those after the step it resumes from."""
    if path is None or not Path(path).exists():
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    Path(path).unlink()
    for r in rows:
        if int(r["step"]) <= step:
            append_row(path, r)


def weights(averaged):
    """The averaged weights as a checkpoint stores them: the network under "ema", the lattice
    model's joint head under "ema_head"."""
    return {("ema" if k == "net" else f"ema_{k}"): m.state_dict() for k, m in averaged.items()}


def train(nets, averaged, groups, loss_fn, validate, *, steps, lr, accum, val_every, ckpt_dir,
          name, meta, generators, resume=True, warmup=3000, clip=1.0, ema_decay=0.999,
          log_every=200, log_path=None, roll_minutes=20.0):
    """Train the modules in ``nets`` to ``steps``, banking their averaged copies at the minimum
    of the validation score.

    ``loss_fn()`` draws one micro-batch and returns its loss; ``accum`` micro-batches make a
    step. ``averaged`` holds the averaged copies, in eval mode. ``validate(averaged)`` returns the
    score a checkpoint is banked by (lower is better) and the readings to log. ``generators`` are
    the generators ``loss_fn`` draws from besides the global ones; their states go into the
    rolling state with the global CPU and CUDA ones. ``meta`` (the configuration) goes into every
    checkpoint.
    """
    out = Path(ckpt_dir)
    out.mkdir(parents=True, exist_ok=True)
    rolling, best_path = out / f"{name}_state.pt", out / f"{name}_best.pt"
    params = [p for net in nets.values() for p in net.parameters()]
    opt = torch.optim.AdamW(groups, lr=lr)
    start, best, best_step, seconds, peak = 0, float("inf"), 0, 0.0, 0.0
    if resume and rolling.exists():
        # onto the CPU: load_state_dict moves weights and optimiser state to their devices, and
        # the random-number states must stay CPU tensors
        rs = torch.load(rolling, map_location="cpu", weights_only=True)
        for k, net in nets.items():
            net.load_state_dict(rs["nets"][k])
        for k, avg in averaged.items():
            avg.load_state_dict(rs["averaged"][k])
        opt.load_state_dict(rs["opt"])
        start, best, best_step = rs["step"], rs["best"], rs["best_step"]
        seconds, peak = rs["seconds"], rs["peak_memory_gb"]
        restore_rng(rs["rng"], generators)
        trim_log(log_path, start)
        print(f"resumed {name} at {start} (best {best:.4f} at {best_step})", flush=True)
    for net in nets.values():
        net.train()
    t0 = last_roll = time.time()

    def cost():
        mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        return seconds + time.time() - t0, max(peak, mem)

    def save_rolling(step):
        s, m = cost()
        save({"nets": {k: net.state_dict() for k, net in nets.items()},
              "averaged": {k: avg.state_dict() for k, avg in averaged.items()},
              "opt": opt.state_dict(), "step": step, "best": best, "best_step": best_step,
              "seconds": s, "peak_memory_gb": m, "rng": rng_state(generators)}, rolling)

    for it in range(start, steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(it, steps, lr, warmup)
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(accum):
            loss = loss_fn() / accum
            loss.backward()
            tot += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        update_average(averaged, nets, ema_decay)
        if it % log_every == 0:
            print(f"LOSS it={it} loss={tot:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if (it + 1) % val_every == 0:
            score, readings = validate(averaged)
            banked = score < best
            if banked:
                best, best_step = score, it + 1
                save({**meta, **weights(averaged), "step": it + 1, "score": score,
                      "readings": readings}, best_path)
            print(f"VAL it={it + 1} " + " ".join(f"{k}={v:.4f}" for k, v in readings.items())
                  + f" best={best:.4f} at {best_step}", flush=True)
            append_row(log_path, {"step": it + 1, **readings, "score": score,
                                  "banked": int(banked)})
            save_rolling(it + 1)
            last_roll = time.time()
        elif time.time() - last_roll >= 60.0 * roll_minutes:
            save_rolling(it + 1)
            last_roll = time.time()
    s, m = cost()
    save({**meta, **weights(averaged), "step": steps, "best_step": best_step,
          "best_score": best, "train_seconds": round(s, 1), "peak_memory_gb": round(m, 2)},
         out / f"{name}_final.pt")
    print(f"TRAIN_DONE {name} best={best:.4f} at {best_step}", flush=True)
    return averaged


# --------------------------------------------------------------- the models

def train_model(model, data, ckpt_dir, name, network=None, dim=768, depth=12, heads=12,
                seq=256, bs=512, micro_bs=64, steps=200000, lr=3e-4, dropout=0.2,
                val_every=5000, val_windows=512, seed=0, resume=True, device="cuda",
                log_path=None):
    """Train one text8 model (the module docstring gives each objective) and return its
    averaged copies."""
    if bs % micro_bs:
        raise ValueError(f"the batch {bs} is not a whole number of micro-batches of {micro_bs}")
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    corpus = Text8(data)
    network = network or NETWORKS[model][0]
    channel = {"graded": Graded(levels=3), "absorbing": ABSORBING}.get(model)
    vocab = channel.vocab if channel else VOC
    size = dict(d=dim, depth=depth, heads=heads, seq=seq, p_drop=dropout)
    accum, windows_per_step = bs // micro_bs, bs
    meta = {"model": model, "network": network, "name": name, "dim": dim, "depth": depth,
            "heads": heads, "seq": seq, "vocab": vocab, "dropout": dropout, "bs": bs,
            "micro_bs": micro_bs, "steps": steps, "lr": lr, "seed": seed,
            "val_every": val_every, "val_windows": val_windows}

    def train_draw():
        return windows(corpus.train, micro_bs, seq, gen).to(device)

    if model == "two-stream":
        net = build(model, network, **size).to(device)
        # the averaged copy is built as a second network (its initialisation draws from the
        # global generator) and then set to the first one's weights
        avg = frozen(build(model, network, **size).to(device))
        avg.load_state_dict(net.state_dict())
        nets, averaged = {"net": net}, {"net": avg}
        groups = [{"params": list(net.parameters()), "weight_decay": 0.01}]
        vbatch = two_stream_validation(corpus.valid, val_windows, seq)
        meta["p_identity"] = P_IDENTITY

        def loss_fn():
            x = train_draw()
            return two_stream_loss(net, x, reveal_order(seq, gen))

        def validate(a):
            r = two_stream_reading(a["net"], vbatch, device)
            return r["identity_bpc"], r
    elif model == "lattice":
        net = build(model, network, **size).to(device)
        head = JointHead(d_net=dim, seq=seq).to(device)
        # as for the two-stream model, the averaged copies are built afresh, then set to the
        # trained modules' weights
        avg_net = frozen(build(model, network, **size).to(device))
        avg_net.load_state_dict(net.state_dict())
        avg_head = frozen(JointHead(d_net=dim, seq=seq).to(device))
        avg_head.load_state_dict(head.state_dict())
        nets, averaged = {"net": net, "head": head}, {"net": avg_net, "head": avg_head}
        groups = [{"params": list(net.parameters()) + list(head.parameters()),
                   "weight_decay": 0.01}]
        accum, sizes = support_sizes(bs, micro_bs)
        windows_per_step = accum * sum(sizes)
        vbatch = lattice_validation(corpus.valid, val_windows, seq)
        meta["supports_per_micro_batch"] = list(sizes)

        def loss_fn():
            return lattice_loss(net, head, corpus.train, sizes, seq, gen, device)

        def validate(a):
            r = lattice_reading(a["net"], a["head"], vbatch, device)
            return (r["prefix_bpc"] + r["masked_bpc"]) / 2, r
    else:
        net = build(model, network, vocab=vocab, **size).to(device)
        nets, averaged = {"net": net}, {"net": frozen(copy.deepcopy(net))}
        groups = decay_groups(net, 0.01)
        vwin = fixed_windows(corpus.valid, val_windows, seq)
        if channel is not None:
            meta["levels"] = channel.levels

            def loss_fn():
                return channel_loss(net, train_draw(), channel, gen)

            def validate(a):
                v = channel_reading(a["net"], vwin, channel, device)
                return v, {"channel_bpc": v}
        elif model == "causal":
            def loss_fn():
                return next_character_loss(net, train_draw())

            def validate(a):
                v = causal_reading(a["net"], vwin, device)
                return v, {"next_character_bpc": v}
        else:
            def loss_fn():
                return masked_loss(net, train_draw(), gen)

            def validate(a):
                v = masked_reading(a["net"], vwin, device)
                return v, {"masked_bpc": v}
    npar = sum(p.numel() for net in nets.values() for p in net.parameters())
    print(f"{name}: the {model} model on the {network} network, {npar / 1e6:.1f}M parameters, "
          f"{windows_per_step} windows of {seq} a step, "
          f"{steps * windows_per_step * seq / len(corpus.train):.0f} passes over the training "
          f"split", flush=True)
    return train(nets, averaged, groups, loss_fn, validate, steps=steps, lr=lr, accum=accum,
                 val_every=val_every, ckpt_dir=ckpt_dir, name=name, meta=meta,
                 generators=(gen,), resume=resume, log_path=log_path)
