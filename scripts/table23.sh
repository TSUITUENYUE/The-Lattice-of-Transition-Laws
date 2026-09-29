#!/usr/bin/env bash
# Table 23: free generation by the 85M lattice model from the all-mask state (256 sequences of 256
# characters, sampled at temperature one) under contiguous, confidence, random, spread and
# word-aware spread at 8, 16, 64 and 256 steps, scored by the dedicated causal model (bpc) and by
# the unigram KL divergence from the corpus at 16 steps, mean and sd over three generation seeds.
# Trains: the lattice model and the dedicated causal model. Reads: the generation.
# Paper compute: the lattice model about 200 L40-hours; the read minutes.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_text8.sh"
{ text8_model_lattice; text8_model_causal; } | train_all
text8_generation_reads | train_all
text8_tables table23
