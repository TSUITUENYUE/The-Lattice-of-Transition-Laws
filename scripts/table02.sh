#!/usr/bin/env bash
# Table 2: an information kernel on MAR-B beside the displacement kernel of Figure 4 (b). The
# information kernel is the mean KL, in bits, of a masked token's head law when one neighbour is
# revealed, on the displacement read's own states (64 ImageNet training images, 8 states each; 8
# drawn values of the revealed token, 32 head draws per token and condition, guidance 2.9). Writes
# to $OUTPUT_ROOT/table02/ both kernels at revealed fraction one half (table02_kernels.csv, the
# left half), every rule's 8-step first-order floor under each kernel beside its FID-50K from
# Table 3 (table02.csv, the right half), and the checks the text states (table02_checks.csv): the
# floors keep their rule order at 16, 32 and 64 steps; the pairs the two kernels order oppositely,
# with their gap on the displacement floor; midpoint's gap below random on each floor.
# Runs: the displacement read (shared with Figure 4), the information read, two floor passes on the
# CPU, and the 38 MAR-B cells of Table 3 (reused when table03.sh ran first).
# Paper compute: the displacement and information reads about 3.5 L40-hours together.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_released.sh"
mar_ready
need "$MAR_DATA/imagenet_train" "IMAGENET_TRAIN=/path/to/ILSVRC2012/train setup/prepare_mar.sh"
mar_kernel_info
mar_cells | train_all
released_tables table02
