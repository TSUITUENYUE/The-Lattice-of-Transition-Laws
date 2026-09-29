#!/usr/bin/env bash
# CIFAR-10: downloads the dataset through torchvision and writes the FID reference, the 50,000
# training images as PNGs in dataset order, against which every CIFAR-10 FID is read (the features
# are cached on first use in $DATA_ROOT/cifar10/fid_cache). Output: $DATA_ROOT/cifar10/
# {cifar-10-batches-py,fid_reference}. A few minutes.
source "$(dirname "$0")/../scripts/_common.sh"
run -m lattice.data.prepare_cifar10 --root "$DATA_ROOT/cifar10"
