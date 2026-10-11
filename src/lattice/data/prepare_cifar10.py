"""Prepare CIFAR-10: the training set and the FID reference.

    python -m lattice.data.prepare_cifar10 --root DATA_ROOT/cifar10

writes under ROOT
  cifar-10-batches-py/     the dataset, downloaded by torchvision
  fid_reference/           the 50,000 training images as 000000.png ... 049999.png, in dataset
                           order; every CIFAR-10 FID is read against this directory
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="where the dataset and the FID reference go")
    args = ap.parse_args(argv)
    import torchvision
    from PIL import Image
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    train = torchvision.datasets.CIFAR10(root=str(root), train=True, download=True)
    torchvision.datasets.CIFAR10(root=str(root), train=False, download=True)
    images = train.data                                   # (50000, 32, 32, 3) uint8
    ref = root / "fid_reference"
    ref.mkdir(exist_ok=True)
    for i, im in enumerate(images):
        Image.fromarray(im).save(ref / f"{i:06d}.png")
    print(f"CIFAR-10 ready under {root}: {len(images)} reference images", flush=True)


if __name__ == "__main__":
    main()
