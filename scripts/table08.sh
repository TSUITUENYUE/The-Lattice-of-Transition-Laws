#!/usr/bin/env bash
# Table 8: ImageNet-256 FID-50K against ADM's reference batch (with sFID, Inception Score,
# precision and recall) of two SiT-B/2 models that differ only in their training distribution over
# levels (a level per patch drawn from the mixed distribution; one level per image), at the
# synchronised corner (249 steps) and at the sequential corner (16 groups of 16 patches, 256
# steps); and the lattice model at the synchronised corner with 16 steps, which moves every
# coordinate as often as the sequential corner does.
# Trains: the two models, 400,000 steps at batch 256. Reads: five cells of 50,000 samples, each in
# four shards, scored with ADM's suite on the CPU.
# Paper compute: on one B200, 0.245 s a step for the lattice model and 0.220 for the other (about
# 27.5 and 24.5 B200-hours); a read about 1.5 B200-hours of sampling and about an hour of CPU.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_images.sh"
need_imagenet
in_models | train_all
printf '%s\n' "lattice synchronised" "lattice sequential" "synchronised synchronised" "synchronised sequential" \
  "lattice synchronised 16" | in_cells
images_tables table08
