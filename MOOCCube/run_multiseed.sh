#!/usr/bin/env bash
set -e
SEEDS=(64) # ...
for seed in "${SEEDS[@]}"; do
  python train_fcih.py --seed "$seed" "$@"
done
