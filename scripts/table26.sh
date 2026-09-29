#!/usr/bin/env bash
# Table 26: forward-pass and selection time per decoding step, medians over the timed steps, the
# device synchronised around each timed region: LLaDA-8B-Base at 64 steps on the first 20 GSM8K
# problems, and the 85M text8 lattice model at 16 steps on 256 windows after a revealed prefix.
# Times depend on the card; the paper's are on one L40. Writes table26_llada.csv and
# table26_text8.csv to $OUTPUT_ROOT/table26/.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_text8.sh"
source "$(dirname "$0")/_released.sh"
text8_table26
llada_ready
llada_timing
released_tables table26_llada table26
