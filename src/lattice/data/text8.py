"""text8: the corpus, its three splits, the windows every reader and trainer takes, and the two
corruption channels of the channel pair.

Characters are coded 0 (space) and 1-26 (a-z); code 27 is the mask. The split is the standard
one: the first 90M characters train, the next 5M validate and the last 5M test (written by
data/prepare_text8.py as train.bin, valid.bin and test.bin). Training draws random windows;
everything reported draws deterministic ones, so two models read the same text.

A channel says what a coordinate's corruption level shows. The absorbing channel has two levels,
the mask and the character. The graded channel has three: the mask, the character's class (space,
vowel or consonant) and the character, so a step can advance a coordinate without committing it.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

V = 27          # a-z and space
MASK = 27
VOC = 28
LN2 = math.log(2.0)


def load_bin(path):
    return np.memmap(path, dtype=np.uint8, mode="r")


class Text8:
    """The corpus, opened once, with the standard 90M/5M/5M split.

    A directory without ``valid.bin`` holds a split whose validation part contains the test
    part; selecting a model on it and reporting on it would be the same measurement twice, so
    opening one raises.
    """

    def __init__(self, root):
        self.root = Path(root)
        if not (self.root / "valid.bin").exists():
            raise FileNotFoundError(
                f"{self.root}/valid.bin missing: prepare the standard 90M/5M/5M split with "
                "setup/prepare_text8.sh.")
        self._cache = {}

    def split(self, name):
        if name not in self._cache:
            self._cache[name] = load_bin(self.root / f"{name}.bin")
        return self._cache[name]

    @property
    def train(self):
        return self.split("train")

    @property
    def valid(self):
        return self.split("valid")

    @property
    def test(self):
        return self.split("test")


def windows(data, n, seq, generator=None):
    """n random windows, for training."""
    idx = torch.randint(0, len(data) - seq, (n,),
                        generator=generator).numpy()
    return torch.from_numpy(
        np.stack([np.asarray(data[i:i + seq]) for i in idx])).long()


def fixed_windows(data, n, seq):
    """n windows spread evenly over the split, for anything reported."""
    starts = np.linspace(0, len(data) - seq - 1, n).astype(np.int64)
    return torch.from_numpy(
        np.stack([np.asarray(data[i:i + seq]) for i in starts])).long()


def mask_batch(x, rates, generator=None, at_least_one=True):
    """Mask each row of x at its own rate.

    With ``at_least_one`` (training) one position per row, drawn uniformly, is masked on top of
    the drawn ones, so no row is left with nothing to score. The masked-ELBO readers pass False:
    a forced position is charged the weight 1/u like any other and would loosen the bound.
    """
    m = torch.rand(x.shape, generator=generator) < rates.view(-1, 1)
    if at_least_one:
        m[torch.arange(len(x)),
          torch.randint(0, x.shape[1], (len(x),), generator=generator)] = True
    xin = x.clone()
    xin[m] = MASK
    return xin, m


def fixed_rates(n, seed):
    """n masking rates drawn uniformly from a fixed seed, clamped to [0.01, 1]."""
    return torch.rand(n, generator=torch.Generator().manual_seed(seed)
                      ).clamp(0.01, 1.0)


VOWELS = "aeiou"


def character_classes():
    """The class of every character: 0 space, 1 consonant, 2 vowel."""
    classes = torch.ones(V, dtype=torch.long)          # consonant
    classes[0] = 0                                     # space
    for c in VOWELS:
        classes[ord(c) - ord("a") + 1] = 2             # vowel
    return classes


class Graded:
    """A channel with ``levels`` values: the mask, the class (only when levels > 2) and the
    character.

    Two levels is the absorbing channel. Three inserts the class level, whose three symbols sit
    above the alphabet and the mask in the vocabulary (codes 28-30), so a rendered state
    announces its own level and a network reads it without being told.
    """

    def __init__(self, levels=3):
        self.levels = levels
        self.classes = character_classes()
        self.symbols = VOC + torch.arange(int(self.classes.max()) + 1)
        self.vocab = VOC + (len(self.symbols) if levels > 2 else 0)

    def render(self, x, level):
        """The observed sequence when coordinate i sits at level[i]: the mask at level 0, the
        class symbol at level 1 (with three levels), the character at the top level."""
        out = torch.full_like(x, MASK)
        if self.levels > 2:
            mid = level == 1
            out[mid] = self.symbols.to(x.device)[self.classes.to(x.device)
                                                 [x[mid]]]
        out[level >= self.levels - 1] = x[level >= self.levels - 1]
        return out

    def draw_levels(self, shape, generator=None):
        """A level per coordinate, uniform over the level set."""
        return torch.randint(0, self.levels, shape, generator=generator)

    def advance(self, level, positions, to=None):
        """Move the named coordinates up one level, or to a named one."""
        level = level.clone()
        step = level.gather(1, positions) + 1
        level.scatter_(1, positions,
                       step.clamp(max=self.levels - 1) if to is None
                       else torch.full_like(step, to))
        return level


ABSORBING = Graded(levels=2)
