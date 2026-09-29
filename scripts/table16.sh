#!/usr/bin/env bash
# Table 16: the 85M lattice model's pair kernel (the entropy decrease at one masked coordinate when
# the other is revealed) with exactly s of the characters between the pair revealed, placed
# uniformly, and the rest of the window masked at the rate: at mask rate 0.5, d = 2, 3, 4 and s = 0
# to d - 1, within a word and across a boundary, 512 pairs per cell, mean over seeds 0-2. The read
# also covers mask rates 0.1, 0.2, 0.3 and 0.7, distances 1, 6, 8, 12 and 16 and s up to 4, the
# kernel that the estimate with the state between of Table 17 reads.
# Trains: the lattice model. Reads: three seeds of the pair kernel with the state between controlled.
# Paper compute: the lattice model about 200 L40-hours; each read minutes.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_text8.sh"
text8_model_lattice | train_all
text8_between_reads | train_all
text8_tables table16
