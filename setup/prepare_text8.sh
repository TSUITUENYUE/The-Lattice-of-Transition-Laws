#!/usr/bin/env bash
# text8 (the first 10^8 characters of an English Wikipedia dump, Mahoney): downloads text8.zip
# (31 MB) and writes the standard 90M/5M/5M split as one byte per character (space 0, a-z 1-26).
# Every text8 model trains on train.bin and is selected on valid.bin; every reported number reads
# test.bin. Output: $DATA_ROOT/text8/{train,valid,test}.bin. About a minute.
source "$(dirname "$0")/../scripts/_common.sh"
run -m lattice.data.prepare_text8 --out "$DATA_ROOT/text8"
