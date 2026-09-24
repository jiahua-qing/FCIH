#!/usr/bin/env bash
set -e
SEEDS=(64 72 80 88 96 104 112 120 128 136)
for seed in "${SEEDS[@]}"; do
  python train_fcih.py --seed "$seed" "$@"
done
