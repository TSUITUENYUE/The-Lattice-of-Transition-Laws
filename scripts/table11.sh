#!/usr/bin/env bash
# Table 11: total correlation of the advanced set (product law less exact chain, 10^-3 bits per
# coordinate) at mask rates 0.3, 0.5 and 0.7 for random and spread sets of 8, 16 and 32 masked
# coordinates, the 85M lattice model, 256 windows per cell, mean and sd over seeds 0-2.
# Trains: the lattice model. Reads: the total-correlation reads (three seeds).
# Paper compute: the lattice model about 200 L40-hours; each read minutes.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_text8.sh"
text8_model_lattice | train_all
text8_sets_reads | train_all
text8_tables table11
