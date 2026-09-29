#!/usr/bin/env bash
# ImageNet-256 latents: every ImageNet-1k training image with ADM's centre crop to 256 x 256,
# encoded by sd-vae-ft-ema (fetched from the Hugging Face hub), and the mirrored crop encoded into
# a second, index-aligned bank. Needs the ILSVRC2012 training images, one directory per synset, at
# $IMAGENET_TRAIN (download them yourself under the ImageNet terms). Output: $DATA_ROOT/imagenet/
# {latents.pt,latents_flip.pt} (about 21 GB each) and their shards. About 5 GPU-hours; extra
# arguments go to the encoder (--shards A:B lets several jobs share the work).
source "$(dirname "$0")/../scripts/_common.sh"
: "${IMAGENET_TRAIN:?set IMAGENET_TRAIN to the ImageNet-1k train/ directory}"
run -m lattice.data.prepare_imagenet --train-dir "$IMAGENET_TRAIN" --out "$DATA_ROOT/imagenet" "$@"
