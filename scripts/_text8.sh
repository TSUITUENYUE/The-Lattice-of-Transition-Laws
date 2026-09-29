# Sourced by the text8 scripts; not a script of its own.
#
# The text8 reads of the paper. Each function prints the commands of one read, one line each, for
# train_all to run (on a GPU of its own each, NUM_GPUS at a time), e.g.
#   { text8_decode_reads; text8_sets_reads; } | train_all
# Every command runs once under its key, so a read shared by several displays is made once. The
# models come from scripts/_text8_models.sh ($T8_LATTICE, $T8_CAUSAL, ...). Every read writes under
# $T8/<read>/, and every text8 display reduces those files (experiments/tables_text8.py):
#   decodes/           decodes of the lattice model after a revealed prefix, every rule, 1-64 steps,
#                      seeds 0-2 (seed 0 also keeps its selected sets)
#   chains/            the same-schedule exact chains at 2-32 steps: every rule at seed 0, random
#                      at seeds 1-15
#   kernel_85M/, kernel_26M/
#                      pair and triplet reads of the 85M and the 26M lattice models, mask rates
#                      0.3/0.5/0.7 (kernel, triplet) and 0.1/0.2 (kernel_low, triplet_low), seeds 0-2
#   estimate/          pairwise and second-order estimates from those reads (no model)
#   sets/ shapes/ overhead/ corners/ readable/ generation/ channels/ channel_kernels/
#   channel_estimate/  one read each, named by what they feed
source "$(dirname "${BASH_SOURCE[0]}")/_text8_models.sh"
T8="$OUTPUT_ROOT/text8"

t8_cmd () { echo "once $1 -m lattice.experiments.${*:2}"; }

# text8_decode_reads [SEEDS]: 256 windows of 256 test characters, 192 revealed, 64 decoded at 1,
# 2, ..., 64 steps under every rule (min_distance: distance 8; midpoint beside separator), with
# each step's mean gap between selected positions;
# seeds 0, 1, 2 unless named (seed 0 keeps its selected sets)
text8_decode_reads () {
  local s extra
  for s in ${1:-0 1 2}; do
    extra=""; [ "$s" = 0 ] && extra="--save-sets"
    t8_cmd "text8_decode_s$s" text8_decode --ckpt "$T8_LATTICE" --data "$T8_DATA" --out "$T8/decodes" \
      --rules contiguous,random,spread,separator,midpoint,dilated,confidence,min_distance \
      --steps 1,2,4,8,16,32,64 --min-distance 8 --separation --seed "$s" $extra
  done
}

# text8_chain_reads: each schedule's decode and its same-schedule exact chain at 2 to 32 steps;
# contiguous, spread, separator, midpoint and confidence at seed 0, random at seeds 0 to 15
text8_chain_reads () {
  local s
  t8_cmd text8_chains_s0 text8_decode --ckpt "$T8_LATTICE" --data "$T8_DATA" --out "$T8/chains" \
    --name text8_chains --rules contiguous,random,spread,separator,midpoint,confidence \
    --steps 2,4,8,16,32 --chain-within-step --seed 0
  for s in $(seq 1 15); do
    t8_cmd "text8_chains_s$s" text8_decode --ckpt "$T8_LATTICE" --data "$T8_DATA" --out "$T8/chains" \
      --name text8_chains --rules random --steps 2,4,8,16,32 --chain-within-step --seed "$s"
  done
}

# text8_pair_reads SIZE CKPT [low]: the pair kernel of one model at mask rates 0.3, 0.5 and 0.7
# (with low: 0.1 and 0.2, a read of its own), 512 pairs per cell, seeds 0-2
text8_pair_reads () {
  local size=$1 ckpt=$2 s
  for s in 0 1 2; do
    if [ "${3:-}" = low ]; then
      t8_cmd "text8_kernel_low_${size}_s$s" text8_dependence kernel --ckpt "$ckpt" --data "$T8_DATA" \
        --out "$T8/kernel_$size" --name kernel_low --rates 0.1,0.2 --seed "$s"
    else
      t8_cmd "text8_kernel_${size}_s$s" text8_dependence kernel --ckpt "$ckpt" --data "$T8_DATA" \
        --out "$T8/kernel_$size" --seed "$s"
    fi
  done
}

# text8_triplet_reads SIZE CKPT [low]: the triplet term, likewise
text8_triplet_reads () {
  local size=$1 ckpt=$2 s
  for s in 0 1 2; do
    if [ "${3:-}" = low ]; then
      t8_cmd "text8_triplet_low_${size}_s$s" text8_dependence triplet --ckpt "$ckpt" --data "$T8_DATA" \
        --out "$T8/kernel_$size" --name triplet_low --rates 0.1,0.2 --seed "$s"
    else
      t8_cmd "text8_triplet_${size}_s$s" text8_dependence triplet --ckpt "$ckpt" --data "$T8_DATA" \
        --out "$T8/kernel_$size" --seed "$s"
    fi
  done
}

# text8_kernel_reads SIZE CKPT: every read the estimates of one model's kernel need
text8_kernel_reads () {
  text8_pair_reads "$1" "$2"; text8_pair_reads "$1" "$2" low
  text8_triplet_reads "$1" "$2"; text8_triplet_reads "$1" "$2" low
}

# t8_files SIZE STEM...: the seed files of the named reads of one model, quoted as one argument
t8_files () {
  local size=$1 stem s out=""; shift
  for stem in "$@"; do for s in 0 1 2; do out="$out $T8/kernel_$size/${stem}_s$s.csv"; done; done
  echo "'${out# }'"
}

# text8_kernel_inputs SIZE: the --structure and --triplet arguments of the estimates
text8_kernel_inputs () {
  echo "--structure $(t8_files "$1" kernel kernel_low) --triplet $(t8_files "$1" triplet triplet_low)"
}

# text8_estimate_reads SIZE: the pairwise and second-order estimates of the rules that read only
# positions, on the decoder's sets selected on the GPU the decodes ran on
text8_estimate_reads () {
  t8_cmd "text8_estimate_$1" text8_estimate fixed "$(text8_kernel_inputs "$1")" --out "$T8/estimate" \
    --name "fixed_$1" --sets-device cuda
}

# text8_between_reads: the pair kernel of the 85M lattice model with s of the coordinates between
# the pair revealed and the rest of the window masked at the rate, s = 0 to 4 (at most d - 1), at
# distances 1, 2, 3, 4, 6, 8, 12 and 16 and mask rates 0.1, 0.2, 0.3, 0.5 and 0.7, 512 pairs per
# cell, seeds 0-2
text8_between_reads () {
  local s
  for s in 0 1 2; do
    t8_cmd "text8_kernel_between_s$s" text8_dependence kernel --ckpt "$T8_LATTICE" --data "$T8_DATA" \
      --out "$T8/kernel_between" --name kernel_between --rates 0.1,0.2,0.3,0.5,0.7 \
      --distances 1,2,3,4,6,8,12,16 --between 0,1,2,3,4 --seed "$s"
  done
}

# text8_between_estimate_reads: the first-order reference and the estimate with the state between,
# from those reads, on the decoder's sets selected on the GPU the decodes ran on
text8_between_estimate_reads () {
  local files="$T8/kernel_between/kernel_between_s0.csv $T8/kernel_between/kernel_between_s1.csv"
  files="$files $T8/kernel_between/kernel_between_s2.csv"
  t8_cmd text8_estimate_between text8_estimate between --structure "'$files'" --out "$T8/estimate" \
    --name between_85M --sets-device cuda
}

# text8_ties_spread_reads: the midpoint rule with ties spread, decoded at 1, 2, ..., 64 steps on the
# windows of text8_decode_reads, with each decode's same-schedule exact chain, seed 0
text8_ties_spread_reads () {
  t8_cmd text8_ties_spread text8_decode --ckpt "$T8_LATTICE" --data "$T8_DATA" --out "$T8/ties_spread" \
    --name text8_ties_spread --rules midpoint_ties_spread --steps 1,2,4,8,16,32,64 --chain-within-step --seed 0
}

# text8_decoded_estimate_reads: the estimates of confidence and min_distance, on the sets the
# seed-0 decodes of text8_decode_reads selected
text8_decoded_estimate_reads () {
  t8_cmd text8_estimate_decoded text8_estimate decoded "$(text8_kernel_inputs 85M)" \
    --sets "$T8/decodes/text8_decode_sets_s0.pt" --out "$T8/estimate" --name decoded_85M
}

# text8_sets_reads: total correlation of random and spread sets of 8, 16, 32 coordinates at mask
# rates 0.3, 0.5, 0.7, 256 windows per cell, seeds 0-2
text8_sets_reads () {
  local s
  for s in 0 1 2; do
    t8_cmd "text8_sets_s$s" text8_dependence sets --ckpt "$T8_LATTICE" --data "$T8_DATA" --out "$T8/sets" \
      --seed "$s"
  done
}

# text8_shapes_reads: complete decodes of 256 characters under five width profiles, contiguous and
# spread, 8 and 16 steps
text8_shapes_reads () {
  t8_cmd text8_shapes text8_decode --ckpt "$T8_LATTICE" --data "$T8_DATA" --out "$T8/shapes" \
    --name text8_shapes --prefix 0 --rules contiguous,spread --steps 8,16 \
    --profiles uniform,wide_first,wide_late,geometric_wide_first,geometric_wide_late --seed 0
}

# text8_overhead_reads: selection and forward-pass time per step at 16 steps, 256 windows, four
# timed decodes per rule, the rules' batch forms
text8_overhead_reads () {
  t8_cmd text8_overhead text8_decode --ckpt "$T8_LATTICE" --data "$T8_DATA" --out "$T8/overhead" \
    --name text8_overhead --rules contiguous,random,spread,separator,midpoint,confidence,min_distance \
    --steps 16 --min-distance 8 --time-steps --time-repeats 4 --select-on-device --seed 0
}

# text8_corner_reads: each model's exact chain after 192 revealed characters over the first 10^6
# test characters, and the masked ELBO on 512 windows, 32 draws
text8_corner_reads () {
  local name ckpt kind
  while read -r name ckpt kind; do
    t8_cmd "text8_corners_$name" text8_corners --ckpt "$ckpt" --kind "$kind" --data "$T8_DATA" \
      --out "$T8/corners" --name "$name"
  done <<EOF
causal $T8_CAUSAL causal
masked $T8_MASKED masked
lattice $T8_LATTICE masked
two_stream $T8_TWO_STREAM two-stream
EOF
}

# text8_readable_reads: the advances of Table 9 under each architecture, 16 draws (seeds 0-15)
text8_readable_reads () {
  local s
  for s in $(seq 0 15); do
    t8_cmd "text8_readable_s$s" text8_readable --causal "$T8_CAUSAL" --bidirectional "$T8_MASKED" \
      --two-stream "$T8_TWO_STREAM" --data "$T8_DATA" --out "$T8/readable" --seed "$s"
  done
}

# text8_generation_reads: free generation by the lattice model at 8, 16, 64 and 256 steps, 256
# sequences, three generation seeds, scored by the dedicated causal model
text8_generation_reads () {
  t8_cmd text8_generation text8_generation --ckpt "$T8_LATTICE" --scorer "$T8_CAUSAL" --data "$T8_DATA" \
    --out "$T8/generation"
}

# text8_channel_reads: the channel pair's complete decodes at 8-64 steps on 512 windows, with the
# graded weights also revealing each character at once, and each model's same-schedule chain
text8_channel_reads () {
  t8_cmd text8_channels text8_channels --absorbing "$T8_ABSORBING" --graded "$T8_GRADED" --data "$T8_DATA" \
    --out "$T8/channels" --chain-within-step
}

# text8_channel_kernel_reads: the pair kernels of the channel pair at mask rates 0.1-0.7, seed 0:
# the absorbing model's, and the graded model's class and character stages
text8_channel_kernel_reads () {
  local rates=0.1,0.2,0.3,0.5,0.7
  t8_cmd text8_channel_kernel_absorbing text8_dependence kernel --ckpt "$T8_ABSORBING" --data "$T8_DATA" \
    --out "$T8/channel_kernels" --name absorbing --rates $rates --seed 0
  t8_cmd text8_channel_kernel_class text8_dependence kernel --ckpt "$T8_GRADED" --data "$T8_DATA" \
    --out "$T8/channel_kernels" --name graded_class --channel graded --stage class --rates $rates --seed 0
  t8_cmd text8_channel_kernel_character text8_dependence kernel --ckpt "$T8_GRADED" --data "$T8_DATA" \
    --out "$T8/channel_kernels" --name graded_character --channel graded --stage character --rates $rates --seed 0
}

# text8_channel_estimate_reads: the channel pair's pairwise bounds (the 85M triplet reads give the
# rates and the corpus crossing fractions), on sets selected on the GPU the decodes ran on
text8_channel_estimate_reads () {
  t8_cmd text8_channel_estimate text8_estimate channel \
    --absorbing "$T8/channel_kernels/absorbing_s0.csv" --class-kernel "$T8/channel_kernels/graded_class_s0.csv" \
    --character-kernel "$T8/channel_kernels/graded_character_s0.csv" \
    --triplet "$(t8_files 85M triplet triplet_low)" \
    --out "$T8/channel_estimate" --sets-device cuda
}

# text8_tables DISPLAY [DIR]: reduce the reads to the rows DISPLAY shows, in $OUTPUT_ROOT/DIR (default DISPLAY)
text8_tables () { run -m lattice.experiments.tables "$1" --results "$T8" --out "$(out_dir "${2:-$1}")"; }

# ---- the text8 parts of displays shared with other families --------------------------------
# Figure 2, token block: the four models at both corners
text8_figure02_tokens () {
  { text8_model_causal; text8_model_masked; text8_model_lattice; text8_model_two_stream; } | train_all
  text8_corner_reads | train_all
  text8_tables figure02_tokens figure02
}

# Figure 5 (a): excess bpc over the left-to-right chain against steps
text8_figure05 () {
  text8_model_lattice | train_all
  text8_decode_reads | train_all
  text8_tables figure05_text8 figure05
}

# Figure 4 (a): the pair kernel at mask rate one half, three seeds
text8_figure04 () {
  text8_model_lattice | train_all
  text8_pair_reads 85M "$T8_LATTICE" | train_all
  text8_tables figure04_text8 figure04
}

# Table 26, text8 rows: selection time against the forward pass at 16 steps
text8_table26 () {
  text8_model_lattice | train_all
  text8_overhead_reads | train_all
  text8_tables table26_text8 table26
}
