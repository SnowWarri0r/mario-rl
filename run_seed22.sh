#!/bin/bash
# 2-2 课程配方的种子实验：严格复现 v1 的条件（4→8→16→30 各 2M，不加载 VecNormalize），
# 只换随机种子跑 3 遍，量"能用 argmax 打的尖解"出现的概率。
# v1 出过 DET 35%，v2（带 pkl + 多两级 + 多 4M）只有 16%/2% —— 所以先搞清是配方稳定产出还是撞上的。
set -eux
SEED=$1
GPU=$2
cd "$(dirname "$0")"
BASE=mario_22ladder_final.zip
for K in 4 8 16 30; do
  OUT=mario_22seed${SEED}_$K
  CUDA_VISIBLE_DEVICES=$GPU MARIO_DEVICE=cuda MARIO_NOOP=$K \
    MARIO_BASE=$BASE MARIO_BASE_VECN=__no_vecnormalize__ \
    MARIO_CKPT_DIR=./checkpoints_22seed${SEED}_$K MARIO_OUT=$OUT \
    venv/bin/python -u train_2_2_noop.py 2000000 48
  BASE=$OUT.zip
done
echo SEED_${SEED}_DONE
