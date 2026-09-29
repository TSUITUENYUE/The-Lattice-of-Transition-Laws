#!/usr/bin/env bash
# Figure 4: kernels measured before decoding. (a) text8: the entropy decrease at one masked
# character when a second one is revealed, inside one word and across a word boundary, at mask
# rate one half, three seeds, 512 pairs per cell (the 85M lattice model). (b) MAR-B: the relative
# displacement of a masked token's prediction after a revealed token is perturbed, against grid
# distance, at revealed fraction one half. Writes figure04_text8.csv and figure04_mar.csv to
# $OUTPUT_ROOT/figure04/.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_text8.sh"
source "$(dirname "$0")/_released.sh"
text8_figure04
mar_ready
mar_kernel
released_tables figure04_mar figure04
