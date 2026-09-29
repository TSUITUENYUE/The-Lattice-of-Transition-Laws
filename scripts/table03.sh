#!/usr/bin/env bash
# Table 3 (left): bits per character of the contiguous, confidence, random, dilated, spread and
# separator rules at 8, 16, 32 and 64 steps on the 64 characters after a revealed 192-character
# prefix, 256 held-out windows, the 85M lattice model; means over seeds 0-2 (only the random rule's
# draws depend on the seed). Table 3 (right): FID-50K of the released MAR-B weights on ImageNet-256
# with only the reveal order changed: raster, MAR's own random order, midpoint, spread and
# low-discrepancy at 8, 16, 32, 64 and 256 steps; nested at 8, 16 and 32; spread for the first
# quarter of the grid, then row by row, at 8 and 16. Random and midpoint at 8 and 16 steps are the
# mean of sampling seeds 0, 1 and 2, every other cell seed 0. Writes table03_text8.csv (left) and
# table03_mar.csv (right) to $OUTPUT_ROOT/table03/.
# Trains: the text8 lattice model. Reads: its decodes (every rule at 1-64 steps, three seeds); 38
# MAR-B cells of 50,000 images each (guidance 2.9 on the linear schedule, temperature 1) and MAR's
# FID.
# Paper compute: the lattice model about 200 L40-hours, each text8 read minutes; a 256-step MAR-B
# cell about 16 L40-hours (12 hours reached about 38,500 images), an 8-step cell about 1 L40S-hour
# (three to a card, about 16,800 images an hour each), cells at 16 to 64 steps in between.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_text8.sh"
source "$(dirname "$0")/_released.sh"
text8_model_lattice | train_all
text8_decode_reads | train_all
text8_tables table03_text8 table03
mar_ready
mar_cells | train_all
released_tables table03_mar table03
