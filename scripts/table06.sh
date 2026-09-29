#!/usr/bin/env bash
# Table 6: bits per character of complete decodes of 512 held-out windows at 8, 16, 32 and 64
# steps by the spread rule: the absorbing model (absorbing, spread), the graded model revealing
# each set's classes and then its characters, both stages charged (graded, class then character),
# and the graded model's weights revealing each character at once (graded, character at once).
# Trains: the absorbing and the graded models of the channel pair. Reads: the channel decodes.
# Paper compute: each model about 64 L40-hours; the read minutes.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_text8.sh"
{ text8_model_absorbing; text8_model_graded; } | train_all
text8_channel_reads | train_all
text8_tables table06
