#!/bin/bash
# 2-2 抖动课程：0-4 → 0-8 → 0-16 → 0-30，每级 2M 步，逐级 warm-start。
# 直接上 0-30 只能从 6% 爬到 17%；抖 2 帧就把原策略腰斩，说明 2→30 之间坡很陡，分级爬看看省不省。
set -eux
cd "$(dirname "$0")"
BASE=mario_22ladder_final.zip
VECN=vecnormalize_22ladder.pkl
for K in 4 8 16 30; do
  OUT=mario_22cur_$K
  CUDA_VISIBLE_DEVICES=1 MARIO_DEVICE=cuda MARIO_NOOP=$K \
    MARIO_BASE=$BASE MARIO_BASE_VECN=$VECN \
    MARIO_CKPT_DIR=./checkpoints_22cur_$K MARIO_OUT=$OUT \
    venv/bin/python -u train_2_2_noop.py 2000000 64
  BASE=$OUT.zip
  VECN=vecnormalize_$OUT.pkl
done
echo CURRICULUM_DONE
