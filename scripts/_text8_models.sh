# Sourced by the text8 scripts; not a script of its own.
#
# Every text8 model of the paper is trained by experiments/train_text8.py (training/text8.py
# states each objective and the checkpoint selection). One function per model prints the command
# that trains it, for train_all or on_gpus to run, e.g.
#   { text8_model_causal; text8_model_lattice; } | train_all
# A model trained once is reused by every script that needs it.
#   text8_model_causal       the dedicated causal model               200,000 steps
#   text8_model_masked       the dedicated masked model               200,000 steps
#   text8_model_two_stream   the two-stream model                     200,000 steps
#   text8_model_lattice      the lattice model                        600,000 steps
#   text8_model_graded       the graded model of the channel pair     200,000 steps
#   text8_model_absorbing    the absorbing model of the channel pair  200,000 steps
#   text8_model_lattice_26m  the 26M lattice model, whose kernel Table 13 reads
# The first six are the paper's 85M configuration: width 768, 12 layers, 12 heads, windows of
# 256, batch 512, dropout 0.2, seed 0. Readers load them at their validation minimum
# ($T8_CAUSAL ... $T8_ABSORBING, the _best.pt files); the 26M lattice model (width 512, 8 layers,
# 8 heads, learned positions, batch 255, no dropout) is read at its last step ($T8_LATTICE_26M).
# Paper compute on one L40: the graded and absorbing models about 1.15 s a step (about 64 hours
# each), the lattice model about 1.2 s a step (about 200 hours).
T8_DATA="$DATA_ROOT/text8"
T8_NETS="$CKPT_ROOT/text8"
T8_TRAIN_LOGS="$OUTPUT_ROOT/train_text8"
[ -f "$T8_DATA/valid.bin" ] || { echo "missing $T8_DATA/valid.bin: run setup/prepare_text8.sh first" >&2; exit 1; }

# text8_train NAME MODEL [FLAGS...]: the command that trains one text8 model as NAME
text8_train () {
  local name=$1 model=$2; shift 2
  echo "once train_text8_$name -m lattice.experiments.train_text8 --model $model --name $name --data $T8_DATA --ckpt-dir $T8_NETS --out $T8_TRAIN_LOGS $*"
}

text8_model_causal () { text8_train causal causal; }
text8_model_masked () { text8_train masked masked; }
text8_model_two_stream () { text8_train two_stream two-stream; }
text8_model_lattice () { text8_train lattice lattice; }
text8_model_graded () { text8_train graded graded; }
text8_model_absorbing () { text8_train absorbing absorbing; }
text8_model_lattice_26m () {
  text8_train lattice_26m lattice --network learned --dim 512 --depth 8 --heads 8 --bs 255 --micro-bs 255 --steps 200000 --lr 3e-4 --dropout 0
}

# the checkpoints the readers load
T8_CAUSAL="$T8_NETS/causal_best.pt"
T8_MASKED="$T8_NETS/masked_best.pt"
T8_TWO_STREAM="$T8_NETS/two_stream_best.pt"
T8_LATTICE="$T8_NETS/lattice_best.pt"
T8_GRADED="$T8_NETS/graded_best.pt"
T8_ABSORBING="$T8_NETS/absorbing_best.pt"
T8_LATTICE_26M="$T8_NETS/lattice_26m_final.pt"
