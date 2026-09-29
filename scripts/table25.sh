#!/usr/bin/env bash
# Table 25: generative perplexity of 512-token samples from the released LLaDA-8B-Base weights
# under the random, confidence, min-distance, midpoint and spread rules at 8 to 64 steps (nucleus
# 0.9), each sample scored in full by GPT-2-large, pooled over 192 samples (three seeds) with a
# bootstrap interval, and the samples' unigram entropy and share of repeated four-token sequences
# (table25.csv); and the comparisons the text reports (table25_comparisons.csv).
# Runs: three seeds of 64 samples per rule and budget (shared with Table 4); nothing trained.
# Compute: 57,600 LLaDA forward passes at batch 2, and the scoring.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_released.sh"
llada_ready
genppl | train_all
released_tables table25
