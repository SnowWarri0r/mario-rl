#!/bin/bash
# 2-2 抖动课程 v2：这次带上 VecNormalize 统计（上一版是在缺 pkl 的情况下跑的），
# 并且在最难档多加两级（0-30 训三程共 6M），因为曲线到 8M 还在爬。
# 级数序列 4→8→16→30→30→30，每级 2M，合计 12M。
set -eux
cd "$(dirname "$0")"
BASE=mario_22ladder_final.zip
VECN=vecnormalize_22ladder.pkl
i=0
for K in 4 8 16 30 30 30; do
  i=$((i+1))
  OUT=mario_22cur2_${i}_$K
  CUDA_VISIBLE_DEVICES=2 MARIO_DEVICE=cuda MARIO_NOOP=$K \
    MARIO_BASE=$BASE MARIO_BASE_VECN=$VECN \
    MARIO_CKPT_DIR=./checkpoints_22cur2_$i MARIO_OUT=$OUT \
    venv/bin/python -u train_2_2_noop.py 2000000 64
  BASE=$OUT.zip
  VECN=vecnormalize_$OUT.pkl
done
echo CURRICULUM_V2_DONE
