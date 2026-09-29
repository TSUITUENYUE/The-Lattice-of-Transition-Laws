#!/usr/bin/env bash
# Table 13: the pairwise estimate from the 26M lattice model's kernel for the schedules and mask
# rates of Table 5, against the 85M lattice model's measured excess over each schedule's own chain,
# at 2 to 32 steps.
# Trains: the lattice model and the 26M lattice model. Reads: the 85M model's same-schedule chains,
# the 26M model's pair and triplet reads (mask rates 0.1-0.7, three seeds), the estimates.
# Paper compute: the lattice model about 200 L40-hours; each read minutes.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_text8.sh"
{ text8_model_lattice; text8_model_lattice_26m; } | train_all
{ text8_chain_reads; text8_kernel_reads 26M "$T8_LATTICE_26M"; } | train_all
text8_estimate_reads 26M | train_all
text8_tables table13
