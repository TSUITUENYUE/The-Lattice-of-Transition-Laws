#!/usr/bin/env bash
# Table 14: bits per character of each rule's sequential chain on the 85M lattice model (every set
# of the schedule revealed one coordinate at a time, left to right within the set), contiguous,
# confidence, random (mean over seeds 0-15), spread and separator at 2 to 32 steps.
# Trains: the lattice model. Reads: the same-schedule chains.
# Paper compute: the lattice model about 200 L40-hours; each read minutes.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_text8.sh"
text8_model_lattice | train_all
text8_chain_reads | train_all
text8_tables table14
