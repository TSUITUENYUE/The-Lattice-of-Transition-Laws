"""The noise levels of a continuous coordinate: knots a_0 > a_1 > ... > a_L in noise variance.

Transition k runs from a_k to a_{k+1}.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Levels:
    """Knots in noise variance, decreasing from a_max to a_min."""

    knots: torch.Tensor

    @staticmethod
    def geometric(a_max: float, a_min: float, transitions: int) -> "Levels":
        i = torch.arange(transitions + 1, dtype=torch.float64)
        return Levels(a_max * (a_min / a_max) ** (i / transitions))

    def __len__(self) -> int:
        """The number of transitions."""
        return len(self.knots) - 1

    def transition(self, k: int) -> tuple[float, float]:
        """(a_k, a_{k+1})."""
        return float(self.knots[k]), float(self.knots[k + 1])

    def condition_number(self) -> float:
        return float(self.knots[0] / self.knots[-1])


def edm_levels(transitions=128, sigma_max=80.0, sigma_min=0.002):
    """EDM's range in sigma, geometric, carried as noise variance."""
    return Levels.geometric(sigma_max ** 2, sigma_min ** 2, transitions)
