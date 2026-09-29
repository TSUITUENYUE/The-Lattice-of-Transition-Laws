#!/usr/bin/env bash
# Table 12: the 85M lattice model's pair kernel at mask rates 0.1, 0.2, 0.3, 0.5 and 0.7,
# distances 1, 2, 4 and 8, pairs inside a word and across a boundary weighted by their corpus
# proportions, seeds pooled: the kernel the estimates of Table 5 read.
# Trains: the lattice model. Reads: its pair and triplet reads (three seeds), the estimate.
# Paper compute: the lattice model about 200 L40-hours; each read minutes.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_text8.sh"
text8_model_lattice | train_all
text8_kernel_reads 85M "$T8_LATTICE" | train_all
text8_estimate_reads 85M | train_all
text8_tables table12
