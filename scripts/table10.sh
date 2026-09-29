#!/usr/bin/env bash
# Table 10: the 85M lattice model's pair kernel (the entropy decrease at one masked coordinate
# when a second at distance d is revealed) at mask rate 0.5, inside one word and across a boundary,
# 512 pairs per cell, mean and sd over seeds 0-2.
# Trains: the lattice model. Reads: its pair reads at mask rates 0.3, 0.5, 0.7 (three seeds).
# Paper compute: the lattice model about 200 L40-hours; each read minutes.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_text8.sh"
text8_model_lattice | train_all
text8_pair_reads 85M "$T8_LATTICE" | train_all
text8_tables table10
