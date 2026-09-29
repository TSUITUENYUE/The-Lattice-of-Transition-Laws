#!/usr/bin/env bash
# Figure 6: intermediate states of the CIFAR-10 lattice model (iid levels) under five schedules,
# one sample at one seed: synchronised (128 steps), the sloped path in raster order at slopes 8 and
# 32 (248 and 608), at slope 32 in spread order (608), and sequential (2063), each at six moments
# of its path, saved as arrays with the level map of every state in
# $OUTPUT_ROOT/images/graded_states/; figure06.csv gives the mean level down each patch column at
# every moment (the caption quotes the fourth).
# Trains: the iid-levels model (shared with Table 7). Reads: five walks of one sample.
# Paper compute: minutes on one GPU once the model is trained.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_images.sh"
need_cifar
px_model iid iid | train_all
px_graded_states | train_all
images_tables figure06
