#!/usr/bin/env bash
# Table 21: for the absorbing and the graded model, the same-schedule sequential chain of the
# decodes of Table 6, the excess of each decode over it, and the pairwise estimate from the model's
# own kernels (the channel walk, each step at its own mask rate, the decode's sets), at 8-64 steps.
# Trains: the absorbing and graded models, and the lattice model (its triplet reads give the walk
# its mask rates and the corpus proportion of pairs across a boundary). Reads: the channel decodes
# with their chains, the channel kernels, the lattice model's triplet reads, the estimate.
# Paper compute: the channel models about 64 L40-hours each, the lattice model about 200; reads minutes.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_text8.sh"
{ text8_model_absorbing; text8_model_graded; text8_model_lattice; } | train_all
{ text8_channel_reads; text8_channel_kernel_reads; text8_triplet_reads 85M "$T8_LATTICE";
  text8_triplet_reads 85M "$T8_LATTICE" low; } | train_all
text8_channel_estimate_reads | train_all
text8_tables table21
