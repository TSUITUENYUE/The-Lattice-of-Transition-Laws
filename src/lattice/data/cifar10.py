"""CIFAR-10 held on the device, in the range EDM's preconditioning assumes, and the class-balanced
labels every CIFAR-10 read draws its samples for (the models are class-conditional)."""
from __future__ import annotations

import torch

SDATA = 0.5          # EDM's assumed data standard deviation
CLASSES = 10


def class_balanced(total, device="cpu"):
    """Labels for a class-balanced sample: total // 10 of each class, the classes in order."""
    return torch.arange(CLASSES, device=device).repeat_interleave(total // CLASSES)


def onehot(labels):
    return torch.eye(CLASSES, device=labels.device)[labels]


class Cifar:
    """Images in [-1, 1] with their labels, resident on the device.

    ``root`` is the directory setup/prepare_cifar10.sh filled (it holds cifar-10-batches-py/).
    """

    def __init__(self, root, device="cuda", train=True):
        import torchvision
        ds = torchvision.datasets.CIFAR10(root=str(root), train=train, download=False)
        self.x = (torch.from_numpy(ds.data).float().permute(0, 3, 1, 2) / 127.5 - 1.0).to(device)
        self.y = torch.tensor(ds.targets, device=device)
        self.eye = torch.eye(CLASSES, device=device)
        self.device = device

    def __len__(self):
        return len(self.x)

    def batch(self, n, generator=None, xflip=True):
        """n training images drawn with replacement, flipped left-right with probability 1/2, and
        their one-hot labels."""
        idx = torch.randint(0, len(self), (n,), generator=generator)
        x = self.x[idx]
        if xflip:
            f = torch.rand(n, generator=generator) < 0.5
            x = torch.where(f.view(-1, 1, 1, 1).to(self.device), x.flip(3), x)
        return x, self.eye[self.y[idx]]
