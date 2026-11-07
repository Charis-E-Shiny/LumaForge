#!/usr/bin/env sh
# Full training sequence. Roughly five hours on a 4-core laptop CPU.
# Each run writes logs/<name>.log and checkpoints/<name>.pt.
set -e
PY="${PYTHON:-python}"
run() { name="$1"; shift; "$PY" scripts/train.py --name "$name" "$@" > "logs/$name.log" 2>&1; tail -1 "logs/$name.log"; }

run teacher --width 24 --iters 5000
run learned --arch learned --width 24 --iters 5000
run student --width 12 --iters 4000 --teacher checkpoints/teacher.pt

# Ablation variants: same width, iterations and seed, one change each.
A="--width 12 --iters 2000 --val-every 500"
run abl_base $A
run abl_no_noise_map $A --no-noise-map
run abl_no_color_head $A --no-color-head
run abl_no_degrade $A --no-degrade
run abl_no_color_loss $A --no-color-loss
run abl_no_classical_denoise $A --no-classical-denoise
run abl_learned $A --arch learned
echo ALL_TRAINING_DONE
