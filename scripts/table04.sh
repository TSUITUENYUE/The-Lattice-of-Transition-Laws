#!/usr/bin/env bash
# Table 4: on the released LLaDA-8B-Base weights, the GSM8K accuracy (greedy, 1,319 problems,
# five-shot) of min-distance (confidence with minimum distance 6) and of the separator constraint
# (confidence with one coordinate per masked run), each minus plain confidence, at 8 to 128 steps
# (paired standard errors); and the min-distance rule's log generative perplexity minus plain
# confidence at 8 to 64 steps, with bootstrap intervals (the samples of Table 25). Also reads the
# kernel whose decay length sets the minimum distance (table04_kernel.csv).
# Runs: 15 GSM8K cells, three generative-perplexity seeds (shared with Table 25), the kernel read.
# Compute: the 15 GSM8K cells an estimated 45 to 60 L40-hours at one problem per decode; the three
# seeds 57,600 forward passes at batch 2.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_released.sh"
llada_ready
llada_kernel
{ gsm8k_cells; genppl; } | train_all
released_tables table04
