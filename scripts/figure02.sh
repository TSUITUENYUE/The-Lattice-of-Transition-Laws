#!/usr/bin/env bash
# Figure 2: both corners of the lattice from one weight set. Pixels: the four CIFAR-10 models
# (lattice with iid levels, lattice with mixed levels, synchronised only, sequential only) with
# their FID-50K at the raster cells of Table 7 and their first samples at a fixed seed at the
# figure's five schedules. Tokens: the four 85M text8 models (dedicated causal, dedicated masked,
# lattice, two-stream), read as an exact chain after 192 revealed characters (AR chain) and as a
# masked-diffusion ELBO (masked ELBO). Writes figure02_pixels.csv and figure02_tokens.csv to
# $OUTPUT_ROOT/figure02/; the sample arrays that figure02_pixels.csv points to stay in
# $OUTPUT_ROOT/images/cifar_cells/.
# Paper compute: each CIFAR-10 model 30 to 40 L40-hours; the text8 lattice model about 200.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_images.sh"
source "$(dirname "$0")/_text8.sh"
images_figure02_pixels
text8_figure02_tokens
