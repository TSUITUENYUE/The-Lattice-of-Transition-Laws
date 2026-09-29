#!/usr/bin/env bash
# Table 5: the pairwise estimate from the 85M lattice model's kernel (the decoder's own sets, each
# step at its own mask rate) against the measured excess bpc over each schedule's own sequential
# chain, for the contiguous, random, spread and separator rules at 2 to 32 steps (the table shows
# 2, 8 and 32 steps, its plot all of them; random measured: the mean over seeds 0-15).
# Trains: the lattice model. Reads: the same-schedule chains, the lattice model's pair and triplet
# reads (mask rates 0.1-0.7, three seeds), the estimates.
# Paper compute: the lattice model about 200 L40-hours; each read minutes.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_text8.sh"
text8_model_lattice | train_all
{ text8_chain_reads; text8_kernel_reads 85M "$T8_LATTICE"; } | train_all
text8_estimate_reads 85M | train_all
text8_tables table05
