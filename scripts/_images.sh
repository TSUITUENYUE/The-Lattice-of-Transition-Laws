# Sourced by the image scripts; not a script of its own.
#
# CIFAR-10. Every model is trained by experiments/train_cifar.py for 60,000 steps at batch 512,
# seed 0, and written to $PX_NETS/ckpt_<name>.pt. The models differ only in their training
# distribution over the lattice (training/measures.py):
#   iid            every patch at an independent level ("iid levels")
#   mixed          a quarter at the synchronised corner, a quarter on the sequential path, half iid
#                  ("mixed levels"); mixed_120k is the same model carried on to 120,000 steps
#   synchronised   one level per image ("synchronised only")
#   sequential     states of the patch-sequential path ("sequential only")
# A cell is one model read along one schedule (experiments/read_cifar.py), named
# MODEL_synchronised_nN, MODEL_sequential_nN or MODEL_slopeK_ORDER_nN, and written to
# $IMG/cifar_reads/<cell>.csv; a model, a cell or a sample file made once is reused by every
# script that needs it.
# ImageNet-256. The two SiT-B/2 models are trained by experiments/train_imagenet.py for 400,000
# steps at batch 256, seed 0, on the plain and the mirrored latent banks, and written to
# $IN_NETS/ckpt_<name>.pt: lattice (a level per patch, the mixed distribution) and synchronised
# (one level per image). A cell is one model read at one corner (experiments/read_imagenet.py),
# written to $IMG/imagenet_reads/<model>_<schedule>[_c<calls>].csv.
# The displays read $IMG (experiments/tables_images.py).
PX_DATA="$DATA_ROOT/cifar10"
PX_NETS="$CKPT_ROOT/cifar"
IN_DATA="$DATA_ROOT/imagenet"
IN_NETS="$CKPT_ROOT/imagenet"
IN_SAMPLES="$OUTPUT_ROOT/imagenet_samples"
ADM="$DATA_ROOT/adm_evaluator"
IMG="$OUTPUT_ROOT/images"

need_cifar () {
  [ -d "$PX_DATA/fid_reference" ] || { echo "missing $PX_DATA/fid_reference: run setup/prepare_cifar10.sh first" >&2; exit 1; }
}

need_imagenet () {
  [ -e "$IN_DATA/latents.pt" ] || { echo "missing $IN_DATA/latents.pt: run setup/prepare_imagenet.sh first" >&2; exit 1; }
  [ -x "$ADM/venv/bin/python" ] || { echo "missing $ADM: run setup/prepare_adm_evaluator.sh first" >&2; exit 1; }
}

# ---- CIFAR-10 ------------------------------------------------------------------------------
# px_model NAME MEASURE [FLAGS...]: the command that trains one CIFAR-10 model
px_model () {
  local name=$1 measure=$2; shift 2
  echo "once train_px_$name -m lattice.experiments.train_cifar --name $name --measure $measure --ckpt-dir $PX_NETS --data-root $PX_DATA $*"
}

# px_models: the commands for the four models of Table 7
px_models () { local m; for m in iid mixed synchronised sequential; do px_model "$m" "$m"; done; }

# px_model_120k: the mixed model carried from 60,000 to 120,000 steps, from its own rolling state
# (every random state restored, so it draws the data an uninterrupted 120,000-step run draws)
px_model_120k () { px_model mixed_120k mixed --steps 120000 --continue-from "$PX_NETS/res_mixed.pt"; }

# px_cell_name MODEL SCHEDULE SLOPE ORDER N: a cell's name
px_cell_name () {
  if [ "$2" = sloped ]; then echo "${1}_slope${3}_${4}_n$5"; else echo "${1}_${2}_n$5"; fi
}

# px_cells: sample and score the cells on stdin, one "MODEL SCHEDULE SLOPE ORDER N" a line.
# All the sampling runs first, each job on a GPU of its own, NUM_GPUS at a time: a sequential cell
# in four shards and every other cell in one (the paper's split, which fixes the noise each sample
# draws); then every cell is scored once. A cell scored earlier is skipped.
px_cells () {
  local -a cells=() jobs=()
  local m s k o n name shards i cell flags
  while read -r m s k o n; do [ -n "$m" ] && cells+=("$m $s $k $o $n"); done
  for cell in "${cells[@]}"; do
    read -r m s k o n <<< "$cell"
    name=$(px_cell_name "$m" "$s" "$k" "$o" "$n")
    [ -f "$OUTPUT_ROOT/.done/read_px_$name" ] && continue
    shards=1; [ "$s" = sequential ] && shards=4
    flags="--ckpt $PX_NETS/ckpt_$m.pt --model $m --name $name --out $IMG/cifar_reads --schedule $s --slope $k --order $o --n $n --shards $shards"
    for ((i = 0; i < shards; i++)); do
      jobs+=("once sample_px_${name}_$i -m lattice.experiments.read_cifar $flags --shard $i --sample-only")
    done
  done
  on_gpus "${jobs[@]+"${jobs[@]}"}"
  for cell in "${cells[@]}"; do
    read -r m s k o n <<< "$cell"
    name=$(px_cell_name "$m" "$s" "$k" "$o" "$n")
    shards=1; [ "$s" = sequential ] && shards=4
    once "read_px_$name" -m lattice.experiments.read_cifar --ckpt "$PX_NETS/ckpt_$m.pt" --model "$m" --name "$name" \
      --out "$IMG/cifar_reads" --schedule "$s" --slope "$k" --order "$o" --n "$n" --shards "$shards" --score-only \
      --reference "$PX_DATA/fid_reference" --fid-cache "$PX_DATA/fid_cache"
  done
}

# px_corner_cells MODEL...: the five raster cells of Figure 2 (synchronised, slopes 1, 4, 16, sequential)
px_corner_cells () {
  local m
  for m in "$@"; do
    echo "$m synchronised 0 raster 50000"
    echo "$m sloped 1 raster 50000"; echo "$m sloped 4 raster 50000"; echo "$m sloped 16 raster 50000"
    echo "$m sequential 0 raster 50000"
  done
}

# px_table_cells: the 28 cells of Table 7, the five raster cells and the two spread cells of each model
px_table_cells () {
  local m
  px_corner_cells iid mixed synchronised sequential
  for m in iid mixed synchronised sequential; do echo "$m sloped 4 spread 50000"; echo "$m sloped 16 spread 50000"; done
}

# px_budget_cells: the mixed model at 120,000 steps, slope 16 in raster and in spread order, over 10,000
# samples and over 50,000
px_budget_cells () {
  local n o
  for n in 10000 50000; do for o in raster spread; do echo "mixed_120k sloped 16 $o $n"; done; done
}

# px_heldout: the mixed model's held-out objective at 60,000 and 120,000 steps
px_heldout () {
  once heldout_px_mixed -m lattice.experiments.heldout_cifar --ckpt "$PX_NETS/ckpt_mixed.pt" "$PX_NETS/ckpt_mixed_120k.pt" \
    --measure mixed --data-root "$PX_DATA" --out "$IMG/heldout" --name heldout_mixed
}

# px_sample_cells MODEL...: the first four samples at seed 0 at the five schedules of Figure 2, as arrays
px_sample_cells () {
  local m
  for m in "$@"; do
    echo "once cells_px_$m -m lattice.experiments.sample_grid cells --ckpt $PX_NETS/ckpt_$m.pt --out $IMG/cifar_cells --name $m"
  done
}

# px_graded_states: the iid-levels model walked along the five schedules of Figure 6, one sample
px_graded_states () {
  echo "once graded_states -m lattice.experiments.sample_grid states --ckpt $PX_NETS/ckpt_iid.pt --out $IMG/graded_states"
}

# ---- ImageNet-256 --------------------------------------------------------------------------
# in_model NAME MEASURE: the command that trains one SiT-B/2
in_model () {
  echo "once train_in_$1 -m lattice.experiments.train_imagenet --name $1 --measure $2 --ckpt-dir $IN_NETS --latents $IN_DATA/latents.pt --flip-bank $IN_DATA/latents_flip.pt"
}

# in_models: the commands for the two models of Table 8
in_models () { in_model lattice mixed; in_model synchronised synchronised; }

# in_cells: sample and score the cells on stdin, one "MODEL SCHEDULE [CALLS]" a line: 50,000
# samples in four shards per cell, each shard on a GPU of its own, NUM_GPUS at a time, then ADM's
# suite on the CPU for every cell. A cell scored earlier is skipped.
in_cells () {
  local -a cells=() jobs=()
  local m s c name cell i calls
  while read -r m s c; do [ -n "$m" ] && cells+=("$m $s ${c:-0}"); done
  for cell in "${cells[@]}"; do
    read -r m s c <<< "$cell"
    name="${m}_$s"; [ "$c" != 0 ] && name="${name}_c$c"
    [ -f "$OUTPUT_ROOT/.done/read_in_$name" ] && continue
    calls=""; [ "$c" != 0 ] && calls="--calls $c"
    for i in 0 1 2 3; do
      jobs+=("once sample_in_${name}_$i -m lattice.experiments.read_imagenet sample --ckpt $IN_NETS/ckpt_$m.pt --schedule $s $calls --sample-dir $IN_SAMPLES/$name --shard $i --num-shards 4")
    done
  done
  on_gpus "${jobs[@]+"${jobs[@]}"}"
  for cell in "${cells[@]}"; do
    read -r m s c <<< "$cell"
    name="${m}_$s"; [ "$c" != 0 ] && name="${name}_c$c"
    once "read_in_$name" -m lattice.experiments.read_imagenet score --sample-dir "$IN_SAMPLES/$name" --adm-dir "$ADM" \
      --out "$IMG/imagenet_reads" --name "$name"
  done
}

# ---- displays ------------------------------------------------------------------------------
# images_tables DISPLAY [DIR]: reduce the reads to the rows DISPLAY shows, in $OUTPUT_ROOT/DIR (default DISPLAY)
images_tables () { run -m lattice.experiments.tables "$1" --results "$IMG" --out "$(out_dir "${2:-$1}")"; }

# Figure 2, pixel block: the four models' FID-50K labels (the raster cells of Table 7) and their
# first samples at a fixed seed at the five schedules the figure shows
images_figure02_pixels () {
  need_cifar
  px_models | train_all
  px_corner_cells iid mixed synchronised sequential | px_cells
  px_sample_cells iid mixed synchronised sequential | train_all
  images_tables figure02_pixels figure02
}
