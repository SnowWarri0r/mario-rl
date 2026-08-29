#!/bin/bash
# 4 帧 vs 8 帧的干净对比：同一份数据蒸两个学生 → 同一套抖动课表微调 → 同口径评测。
# 差异只剩叠帧数。用法: bash run_stack_bridge.sh <4|8> <gpu>
set -eux
cd "$(dirname "$0")"
K=$1; GPU=$2
# ① 蒸馏：把梯子专家的通关技能灌进 K 帧学生
CUDA_VISIBLE_DEVICES=$GPU MARIO_STACK=$K venv/bin/python -u distill_stack.py 30 mario_22stack$K
# ② 蒸完先验一次：K 帧学生在 noop=0 下应该接近老师(66-78%)，不然说明桥没搭上
CUDA_VISIBLE_DEVICES=$GPU MARIO_STACK=$K venv/bin/python -u eval_noop_audit.py \
  "models22:mario_22stack$K.zip" 200 24
# ③ 抖动课程微调：0-4 → 0-8 → 0-16 → 0-30，每级 1M
BASE=mario_22stack$K.zip
for J in 4 8 16 30; do
  OUT=mario_22stack${K}_ft$J
  CUDA_VISIBLE_DEVICES=$GPU MARIO_DEVICE=cuda MARIO_STACK=$K MARIO_NOOP=$J \
    MARIO_BASE=$BASE MARIO_BASE_VECN=vecnormalize_$OUT.pkl \
    MARIO_CKPT_DIR=./checkpoints_22stack${K}_ft$J MARIO_OUT=$OUT \
    venv/bin/python -u train_2_2_noop.py 1000000 48
  BASE=$OUT.zip
done
echo STACK_${K}_DONE
