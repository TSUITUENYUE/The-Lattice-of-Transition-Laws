#!/usr/bin/env bash
# Table 9: the loss of advances from lattice states under the law each architecture expresses (the
# causal model at prefix states, the bidirectional (dedicated masked) model as a product law, the
# two-stream model as a sequential chain): prefix width 1, interior width 1 and interior width 8,
# mean and sd over 16 draws (table09.csv); and the paired excess of the two-stream over the
# bidirectional model at interior width 1 that the caption quotes (table09_paired.csv).
# Trains: the dedicated causal, the dedicated masked and the two-stream models. Reads: 16 draws.
# Paper compute: each read minutes.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_text8.sh"
{ text8_model_causal; text8_model_masked; text8_model_two_stream; } | train_all
text8_readable_reads | train_all
text8_tables table09
