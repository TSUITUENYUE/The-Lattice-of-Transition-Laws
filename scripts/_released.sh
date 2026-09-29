# Sourced by the scripts of the experiments on released weights (LLaDA-8B-Base, MAR-B, SkyReels-V2);
# not a script of its own.
#
# Nothing is trained. setup/prepare_llada.sh, setup/prepare_mar.sh and setup/prepare_video.sh fetch the
# weights, code and data; MAR-B and SkyReels-V2 run in environments of their own, which those scripts
# build ($MAR_PYTHON, $VIDEO_PYTHON), and LLaDA in the main one. Every experiment writes under
# $RELEASED/<experiment>/ and runs once: a finished run is reused by every script that needs it, and an
# interrupted one resumes where it stopped. HumanEval candidates execute only in a sandbox, chosen by
# SANDBOX: bwrap (default) or unshare, or docker or podman with SANDBOX_IMAGE=sha256:<digest> of a local
# image holding Python.
LLADA_DATA="$DATA_ROOT/llada"
MAR_DATA="$DATA_ROOT/mar"
VIDEO_DATA="$DATA_ROOT/video"
RELEASED="$OUTPUT_ROOT/released"
LLADA="$LLADA_DATA/LLaDA-8B-Base"
MAR="$MAR_DATA/mar"
MAR_PYTHON="$MAR_DATA/venv/bin/python"
SKYREELS="$VIDEO_DATA/SkyReels-V2"
SKYREELS_MODEL="$VIDEO_DATA/SkyReels-V2-DF-1.3B-540P"
VIDEO_PYTHON="$VIDEO_DATA/venv/bin/python"
SANDBOX="${SANDBOX:-bwrap}"
LLADA_RULES="confidence confidence_min_distance confidence_separator"
LLADA_BUDGETS="8 16 32 64 128"

# released_tables DISPLAY [DIR]: reduce the runs to the rows DISPLAY shows, in $OUTPUT_ROOT/DIR (default DISPLAY)
released_tables () { run -m lattice.experiments.tables "$1" --results "$RELEASED" --out "$(out_dir "${2:-$1}")"; }

need () { [ -e "$1" ] || { echo "missing $1: run $2 first" >&2; exit 1; }; }

# ---- LLaDA-8B-Base -------------------------------------------------------------------------------
llada_ready () {
  need "$LLADA/config.json" setup/prepare_llada.sh
  need "$LLADA_DATA/gsm8k/test.json" setup/prepare_llada.sh
}

# gsm8k RULE STEPS [TEMPERATURE]: the command for one GSM8K cell (1,319 problems), greedy unless a
# temperature is given
gsm8k () {
  local rule=$1 steps=$2 t=${3:-0} tag=""
  [ "$t" = 0 ] || tag="_t$t"
  echo "once llada_gsm8k_${rule}_s${steps}${tag} -m lattice.experiments.llada_gsm8k --model $LLADA --data $LLADA_DATA/gsm8k --out $RELEASED/gsm8k --rule $rule --steps $steps --temperature $t --min-distance 6 --seed 1234"
}

# gsm8k_cells: the commands for the 15 greedy cells of Table 4 (3 rules x 8 to 128 steps)
gsm8k_cells () {
  local rule s
  for rule in $LLADA_RULES; do for s in $LLADA_BUDGETS; do gsm8k "$rule" "$s"; done; done
}

# genppl: the commands for the three seeds of generative perplexity (generation, then scoring by
# GPT-2-large in a separate process), one GPU each
genppl () {
  local s
  for s in 0 1 2; do
    echo "once llada_genppl_generate_s$s -m lattice.experiments.llada_genppl generate --model $LLADA --out $RELEASED/genppl --seed $s && once llada_genppl_score_s$s -m lattice.experiments.llada_genppl score --scorer $LLADA_DATA/gpt2-large --out $RELEASED/genppl --seed $s"
  done
}

# llada_kernel: the kernel read on WikiText-103 text whose decay length sets the minimum distance
llada_kernel () {
  once llada_kernel -m lattice.experiments.llada_kernel --model "$LLADA" --corpus "$LLADA_DATA/wikitext103_test.txt" \
    --out "$RELEASED/llada_kernel"
}

# humaneval RULE STEPS: the command that generates one HumanEval cell (164 problems; nothing runs)
humaneval () {
  echo "once llada_humaneval_$1_s$2 -m lattice.experiments.llada_humaneval generate --model $LLADA --data $LLADA_DATA/HumanEval.jsonl.gz --out $RELEASED/humaneval --rule $1 --steps $2 --min-distance 6 --seed 1234"
}

# humaneval_evaluate: check the sandbox, then execute every generated cell's programs in it (CPU)
humaneval_evaluate () {
  local -a box=(--sandbox "$SANDBOX")
  [ -z "${SANDBOX_IMAGE:-}" ] || box+=(--image "$SANDBOX_IMAGE")
  once "llada_humaneval_sandbox_$SANDBOX" -m lattice.experiments.llada_humaneval check-sandbox \
    --data "$LLADA_DATA/HumanEval.jsonl.gz" --out "$RELEASED/humaneval" "${box[@]}"
  local rule s
  for rule in $LLADA_RULES; do for s in $LLADA_BUDGETS; do
    once "llada_humaneval_evaluate_${rule}_s$s" -m lattice.experiments.llada_humaneval evaluate \
      --data "$LLADA_DATA/HumanEval.jsonl.gz" --out "$RELEASED/humaneval" --rule "$rule" --steps "$s" "${box[@]}"
  done; done
}

# llada_timing: the LLaDA rows of Table 26, the first 20 GSM8K problems at 64 steps under six rules,
# decoded in turn on one GPU with every step's forward pass and selection timed
llada_timing () {
  local rule
  for rule in random confidence confidence_min_distance confidence_separator midpoint spread; do
    once "llada_timing_$rule" -m lattice.experiments.llada_gsm8k --model "$LLADA" --data "$LLADA_DATA/gsm8k" \
      --out "$RELEASED/llada_timing" --rule "$rule" --steps 64 --limit 20 --min-distance 6 --seed 1234
  done
}

# ---- MAR-B ------------------------------------------------------------------------------------------
mar_ready () {
  need "$MAR/pretrained_models/mar/mar_base/checkpoint-last.pth" setup/prepare_mar.sh
  need "$MAR_PYTHON" setup/prepare_mar.sh
}

# mar_cells: the commands for the 38 FID-50K cells of Table 3, one GPU each: five reveal orders (raster,
# random, midpoint, spread, low-discrepancy) at 8 to 256 steps, sampling seed 0; random and midpoint at 8 and 16
# steps with seeds 1 and 2 as well; nested at 8, 16 and 32 steps; spread-then-raster at 8 and 16 steps
mar_cells () {
  local rule s k
  for rule in raster random midpoint spread low_discrepancy; do for s in 8 16 32 64 256; do
    echo "PYTHON=$MAR_PYTHON once mar_${rule}_s$s -m lattice.experiments.mar_decode --mar $MAR --rule $rule --num-iter $s --out $RELEASED/mar"
  done; done
  for rule in random midpoint; do for s in 8 16; do for k in 1 2; do
    echo "PYTHON=$MAR_PYTHON once mar_${rule}_s${s}_seed$k -m lattice.experiments.mar_decode --mar $MAR --rule $rule --num-iter $s --seed $k --out $RELEASED/mar"
  done; done; done
  for s in 8 16 32; do
    echo "PYTHON=$MAR_PYTHON once mar_nested_s$s -m lattice.experiments.mar_decode --mar $MAR --rule nested --num-iter $s --out $RELEASED/mar"
  done
  for s in 8 16; do
    echo "PYTHON=$MAR_PYTHON once mar_spread_raster_s$s -m lattice.experiments.mar_decode --mar $MAR --rule spread_raster --num-iter $s --out $RELEASED/mar"
  done
}

# mar_kernel: the read of Figure 4 (b) on 64 ImageNet training images (setup/prepare_mar.sh links them)
mar_kernel () {
  need "$MAR_DATA/imagenet_train" "IMAGENET_TRAIN=/path/to/ILSVRC2012/train setup/prepare_mar.sh"
  PYTHON="$MAR_PYTHON" once mar_kernel -m lattice.experiments.mar_kernel --mar "$MAR" \
    --imagenet "$MAR_DATA/imagenet_train" --out "$RELEASED/mar_kernel"
}

# mar_kernel_info: the reads of Table 2. The information kernel on the displacement read's own states (the
# information mode replays that read's random draws), then every rule's first-order floor under each kernel
mar_kernel_info () {
  mar_kernel
  PYTHON="$MAR_PYTHON" once mar_kernel_information -m lattice.experiments.mar_kernel --mar "$MAR" \
    --imagenet "$MAR_DATA/imagenet_train" --out "$RELEASED/mar_kernel_info" --information --draws 8 --head-samples 32 --cfg 2.9
  PYTHON="$MAR_PYTHON" once mar_kernel_floors -m lattice.experiments.mar_kernel --mar "$MAR" \
    --imagenet "$MAR_DATA/imagenet_train" --floor "$RELEASED/mar_kernel/kernel.csv" --out "$RELEASED/mar_kernel"
  PYTHON="$MAR_PYTHON" once mar_kernel_information_floors -m lattice.experiments.mar_kernel --mar "$MAR" \
    --imagenet "$MAR_DATA/imagenet_train" --floor "$RELEASED/mar_kernel_info/information.csv" --out "$RELEASED/mar_kernel_info"
}

# ---- SkyReels-V2 ------------------------------------------------------------------------------------
video_ready () {
  need "$SKYREELS_MODEL/config.json" setup/prepare_video.sh
  need "$VIDEO_PYTHON" setup/prepare_video.sh
}

# video_cells: the commands for the 15 rollouts of Table 28 (3 prompts x memory depths 5, 17, 25, 33, 57)
video_cells () {
  local p ov
  for p in 0 1 2; do for ov in 5 17 25 33 57; do
    echo "PYTHON=$VIDEO_PYTHON once video_p${p}_ov$ov -m lattice.experiments.video_memory generate --skyreels $SKYREELS --model $SKYREELS_MODEL --prompt-index $p --overlap $ov --out $RELEASED/video/cells"
  done; done
}

# video_score: the four VBench dimensions declared before decoding, for the 15 rollouts (Table 28)
video_score () {
  PYTHON="$VIDEO_PYTHON" once video_score -m lattice.experiments.video_memory score --videos "$RELEASED/video/cells" \
    --out "$RELEASED/video"
}

# video_kernel: the kernel between frames that Appendix D quotes beside Table 28, read on the model's own
# sampling states
video_kernel () {
  PYTHON="$VIDEO_PYTHON" once video_kernel -m lattice.experiments.video_kernel --skyreels "$SKYREELS" \
    --model "$SKYREELS_MODEL" --out "$RELEASED/video/kernel"
}
