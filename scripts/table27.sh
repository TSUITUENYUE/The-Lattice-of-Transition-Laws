#!/usr/bin/env bash
# Table 27: HumanEval pass@1 (164 problems, greedy) on the released LLaDA-8B-Base weights under
# plain confidence, min-distance and the separator constraint at 8 to 128 steps, and GSM8K at 64
# steps under the authors' sampling (temperature 1) for plain confidence and min-distance
# (table27.csv); the paired comparisons, with McNemar tests (table27_paired.csv); and plain
# confidence on GSM8K at 256 steps, the authors' setting (table27_gsm8k_256.csv).
# Runs: 15 HumanEval cells (generation on the GPU, then every program in a sandbox on the CPU), 3
# GSM8K cells.
# Compute: 122,016 forward passes for HumanEval; the GSM8K cells as in Table 4 (256 steps: twice a
# 128-step cell).
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_released.sh"
llada_ready
{ for rule in $LLADA_RULES; do for s in $LLADA_BUDGETS; do humaneval "$rule" "$s"; done; done
  gsm8k confidence 64 1; gsm8k confidence_min_distance 64 1; gsm8k confidence 256; } | train_all
humaneval_evaluate
released_tables table27
