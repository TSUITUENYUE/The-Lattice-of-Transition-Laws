#!/usr/bin/env bash
# Figure 5: schedule quality against the number of decoding steps. (a) text8: bits per character
# of the 85M lattice model's decodes after a revealed prefix, the six rules of Table 3 at 1 to 64
# steps, over the exact left-to-right chain. (b) MAR-B: FID-50K over our read of MAR-B's own random
# order at 256 steps, five reveal orders at 8 to 256 steps (the cells of Table 3; random and
# midpoint at 8 and 16 steps average three seeds). Writes figure05_text8.csv and figure05_mar.csv
# to $OUTPUT_ROOT/figure05/.
# Paper compute: the lattice model about 200 L40-hours; a 256-step MAR cell about 16.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_text8.sh"
source "$(dirname "$0")/_released.sh"
text8_figure05
mar_ready
mar_cells | train_all
released_tables figure05_mar figure05
