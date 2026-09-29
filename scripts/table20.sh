#!/usr/bin/env bash
# Table 20: bits per character of complete decodes (256 characters, no prefix, 256 held-out
# windows) under five width profiles (uniform; wide first and wide late, widths proportional to
# (t+1)^-1 and (t+1)^+1; geometric, widths scaled by 2/3 or 3/2 per step), contiguous and spread at
# 8 and 16 steps, the 85M lattice model, one deterministic read.
# Trains: the lattice model. Reads: the width-profile decodes.
# Paper compute: the lattice model about 200 L40-hours; the read minutes.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_text8.sh"
text8_model_lattice | train_all
text8_shapes_reads | train_all
text8_tables table20
