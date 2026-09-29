#!/usr/bin/env bash
# Table 19: at 16 steps (width 4) on the protocol of Table 18, bpc and separation (the mean gap
# between consecutive selected positions) of confidence, min-distance (confidence with a minimum
# distance of 8), random and spread (means over seeds 0-2), and the pairwise estimate on each rule's
# own sets (the confidence rules: the sets their seed-0 decode selected, after the decode; random
# and spread: the decoder's sets, before it).
# Trains: the lattice model. Reads: the decodes of Table 3 (three seeds), the pair and triplet reads
# (mask rates 0.1-0.7, three seeds), the estimates.
# Paper compute: the lattice model about 200 L40-hours; each read minutes.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_text8.sh"
text8_model_lattice | train_all
{ text8_decode_reads; text8_kernel_reads 85M "$T8_LATTICE"; } | train_all
{ text8_estimate_reads 85M; text8_decoded_estimate_reads; } | train_all
text8_tables table19
