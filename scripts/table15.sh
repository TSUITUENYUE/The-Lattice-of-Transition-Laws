#!/usr/bin/env bash
# Table 15: the second-order estimate (the pairwise estimate plus the triplet term on unscreened
# consecutive triples, each step at its own mask rate) against the measured excess over each
# schedule's own chain, for contiguous, random, spread and separator at 2 to 32 steps, and for the
# confidence rule, estimated after its seed-0 decode on the sets it selected.
# Trains: the lattice model. Reads: the decodes at seed 0 (the confidence decode's sets), the
# same-schedule chains, the lattice model's pair and triplet reads, the estimates.
# Paper compute: the lattice model about 200 L40-hours; each read minutes.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_text8.sh"
text8_model_lattice | train_all
{ text8_decode_reads 0; text8_chain_reads; text8_kernel_reads 85M "$T8_LATTICE"; } | train_all
{ text8_estimate_reads 85M; text8_decoded_estimate_reads; } | train_all
text8_tables table15
