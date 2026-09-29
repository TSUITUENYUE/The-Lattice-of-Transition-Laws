#!/usr/bin/env bash
# Table 24: min-distance against plain confidence on the samples of Table 25, at 8 to 64 steps: the
# pooled difference in log generative perplexity; the rule's coefficient when each sample's log
# perplexity is regressed on the rule, its unigram entropy and its share of repeated four-token
# sequences in the scorer's tokens, with a 95% bootstrap interval (4,000 resamples, seed 0); and the
# pooled difference among the samples whose share lies below the median of both rules' samples.
# Runs: the three generative-perplexity seeds of Table 25 (reused when table25.sh ran first);
# nothing trained.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_released.sh"
llada_ready
genppl | train_all
released_tables table24
