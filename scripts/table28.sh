#!/usr/bin/env bash
# Table 28: memory depth on the released SkyReels-V2-DF-1.3B-540P weights at eight times their
# training window (773 frames): the decoding steps of depths 5, 17 (as shipped), 25, 33 and 57
# frames, from the segment plan, and the four VBench dimensions declared before decoding (subject
# consistency, background consistency, motion smoothness, imaging quality) over the whole rollout:
# the shipped depth's mean over the three prompts, every other depth's paired difference from it
# with its paired standard error (table28.csv); and the kernel between frames that Appendix D
# quotes beside the table, read on the model's own sampling states (table28_kernel.csv).
# Runs: 15 rollouts (3 prompts x 5 depths, seed 0), their VBench scores, the frame kernel; nothing
# trained.
# Compute: about 3 minutes per segment on one RTX 4090 (9 to 18 segments per rollout); the kernel
# 19 minutes.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_released.sh"
video_ready
video_cells | train_all
video_score
video_kernel
released_tables table28
