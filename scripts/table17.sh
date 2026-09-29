#!/usr/bin/env bash
# Table 17: the separator rule, the midpoint rule and the midpoint rule with ties spread on the 85M
# lattice model at 8, 16 and 32 steps (the windows of Table 18, 192 revealed, 64 decoded). The
# first-order reference and the estimate with the state between walk the decoder's sets over the
# kernel reads of Table 16 (mask rates 0.1 to 0.7, each step at its own rate); the measured excess
# is each decode minus its schedule's own chain (separator and midpoint: the chains of Table 14;
# midpoint with ties spread: its own decode and chain at 1 to 64 steps, seed 0).
# Trains: the lattice model. Reads: the kernel with the state between (three seeds), the chains,
# the decode and chain of the midpoint rule with ties spread, then the estimates on the CPU.
# Paper compute: the lattice model about 200 L40-hours; each read minutes.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_text8.sh"
text8_model_lattice | train_all
{ text8_between_reads; text8_chain_reads; text8_ties_spread_reads; } | train_all
text8_between_estimate_reads | train_all
text8_tables table17
