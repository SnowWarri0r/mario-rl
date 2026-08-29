#!/bin/bash
# W2/W3 抖动重训：W1 pilot 已验证有效（老师平均 40.5%→52.8%，1-3 从 0%→59%）。
# 等种子实验腾出 CPU 再开。
set -eux
cd "$(dirname "$0")"
while pgrep -f "train_2_2_noop.py 2000000 48" > /dev/null; do sleep 120; done
CUDA_VISIBLE_DEVICES=3 MARIO_DEVICE=cuda MARIO_NOOP=30 MARIO_OUT=mario_w2noop \
  MARIO_BASE=mario_w2land_final.zip MARIO_BASE_VECN=vecnormalize_w2land.pkl \
  venv/bin/python -u train_world_noop.py w2 5000000 48 &
CUDA_VISIBLE_DEVICES=4 MARIO_DEVICE=cuda MARIO_NOOP=30 MARIO_OUT=mario_w3noop \
  MARIO_BASE=mario_w3_final.zip MARIO_BASE_VECN=vecnormalize_w3.pkl \
  venv/bin/python -u train_world_noop.py w3 5000000 48 &
wait
echo W23_DONE
