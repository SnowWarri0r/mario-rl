#!/bin/bash
# 2-2 换 skip=2：一格宽的鱼缝要帧级精度，每个动作按 4 帧可能不够细。
# warm-start 自课程最好的那个（已经是反应式的），带抖动。等在跑的两条腾出 CPU 再开。
set -eux
cd "$(dirname "$0")"
while pgrep -f "train_world1_noop.py" > /dev/null || pgrep -f "train_2_2_noop.py 5000000 64" > /dev/null; do
  sleep 60
done
CUDA_VISIBLE_DEVICES=1 MARIO_DEVICE=cuda MARIO_NOOP=30 MARIO_SKIP=2 \
  MARIO_BASE=mario_22cur_30.zip MARIO_BASE_VECN=vecnormalize_mario_22cur_30.pkl \
  MARIO_CKPT_DIR=./checkpoints_22skip2 MARIO_OUT=mario_22skip2 \
  venv/bin/python -u train_2_2_noop.py 8000000 64
echo SKIP2_DONE
