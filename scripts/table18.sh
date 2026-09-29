#!/usr/bin/env bash
# Table 18: bits per character at 64, 32, 16, 8, 4 and 2 steps for the spread, random and
# contiguous rules on the protocol of Table 3 (64 characters after a revealed 192-character prefix,
# 256 held-out windows, the 85M lattice model), mean and sd over seeds 0-2.
# Trains: the lattice model. Reads: the decodes of Table 3 (three seeds).
# Paper compute: the lattice model about 200 L40-hours; each read minutes.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_text8.sh"
text8_model_lattice | train_all
text8_decode_reads | train_all
text8_tables table18
