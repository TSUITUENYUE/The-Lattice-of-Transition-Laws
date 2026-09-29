# Sourced by every script in this directory; not a script of its own.
#
# Every path, the interpreter and the GPU count are fixed here and nowhere else.
# Override any of them from the environment, e.g.
#   DATA_ROOT=/data OUTPUT_ROOT=/scratch/out NUM_GPUS=4 bash scripts/table05.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
# datasets and reference statistics, prepared once by setup/*.sh
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
# checkpoints are shared between scripts; every table or figure writes its own directory,
# $OUTPUT_ROOT/tableNN or $OUTPUT_ROOT/figureNN (out_dir below)
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs}"
CKPT_ROOT="${CKPT_ROOT:-$OUTPUT_ROOT/checkpoints}"
NUM_GPUS="${NUM_GPUS:-1}"
export DATA_ROOT OUTPUT_ROOT CKPT_ROOT PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT_ROOT" "$CKPT_ROOT"

# run ARGS...: the interpreter on ARGS, echoing the command first.
run () { echo "+ $PYTHON $*" >&2; "$PYTHON" "$@"; }

# on_gpus CMD...: run each quoted command on a GPU of its own, NUM_GPUS at a
# time, and fail if any of them fails. With NUM_GPUS=1 the commands run in turn.
on_gpus () {
  local i=0 rc=0 pid cmd
  local -a pids=()
  for cmd in "$@"; do
    ( export CUDA_VISIBLE_DEVICES=$((i % NUM_GPUS)); eval "$cmd" ) &
    pids+=("$!")
    i=$((i + 1))
    if (( ${#pids[@]} == NUM_GPUS )); then
      for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
      pids=()
    fi
  done
  for pid in "${pids[@]+"${pids[@]}"}"; do wait "$pid" || rc=1; done
  return "$rc"
}

# train_all: run every command on stdin (one per line), each on a GPU of its own, NUM_GPUS at a time.
train_all () {
  local -a jobs=()
  local line
  while IFS= read -r line; do [ -n "$line" ] && jobs+=("$line"); done
  on_gpus "${jobs[@]+"${jobs[@]}"}"
}

# once KEY ARGS...: run the interpreter on ARGS unless a run under KEY already finished. Scripts
# that share an experiment share its KEY, so the second one reuses the first one's outputs.
once () {
  local key=$1; shift
  local stamp="$OUTPUT_ROOT/.done/$key"
  if [ -f "$stamp" ]; then echo "= $key: finished earlier, reused" >&2; return 0; fi
  run "$@"
  mkdir -p "$OUTPUT_ROOT/.done"
  date > "$stamp"
}

# out_dir NAME: the directory a table or figure writes to (tableNN, figureNN), created on demand.
out_dir () { mkdir -p "$OUTPUT_ROOT/$1"; echo "$OUTPUT_ROOT/$1"; }
