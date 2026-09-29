#!/usr/bin/env bash
# Table 22: pair kernels of the channel pair at mask rate 0.7, distances 1, 2, 4, 8, within one
# word / across a boundary, 512 pairs per cell, seed 0: the absorbing model's character kernel, the
# graded model's class kernel and its character kernel given the classes.
# Trains: the absorbing and the graded models. Reads: the channel kernels (mask rates 0.1-0.7).
# Paper compute: each model about 64 L40-hours; the reads minutes.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_text8.sh"
{ text8_model_absorbing; text8_model_graded; } | train_all
text8_channel_kernel_reads | train_all
text8_tables table22
