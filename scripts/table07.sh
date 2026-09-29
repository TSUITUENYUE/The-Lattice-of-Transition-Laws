#!/usr/bin/env bash
# Table 7: CIFAR-10 FID-50K of four models that differ only in their training distribution
# (iid levels, mixed levels, synchronised only, sequential only) at seven schedules: synchronised
# (128 steps), the sloped path at slope 1 (143), at slopes 4 and 16 in raster and in spread order
# (188 and 368), and sequential (2063), and the mixed model carried on to 120,000 steps at slope 16
# (table07.csv); and the readings the appendix quotes with the table (table07_budget.csv): the
# mixed model's held-out objective at 60,000 and 120,000 steps, and its two slope-16 cells at
# 120,000 steps over 10,000 samples and over 50,000.
# Trains: the four models (60,000 steps at batch 512) and the mixed model carried on to 120,000.
# Reads: 28 cells of 50,000 samples (a sequential cell in four shards), four cells at 120,000 steps
# (10,000 and 50,000 samples), and the held-out objective at the two budgets.
# Paper compute: 1.8 to 2.5 s a training step on one L40 (30 to 40 L40-hours for 60,000 steps);
# one model's seven cells about 64 L40-hours, the sequential cell alone 32 to 46.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_images.sh"
need_cifar
px_models | train_all
px_model_120k | train_all
px_table_cells | px_cells
px_budget_cells | px_cells
px_heldout
images_tables table07
