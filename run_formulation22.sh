#!/bin/bash
# 2-2 换问题表述：三条臂同课表对比。
#   ff4  前馈叠 4 帧（对照组，跟另两条同样从零训、同样课表）
#   ff8  前馈叠 8 帧（看更长一段的鱼运动轨迹）
#   lstm RecurrentPPO CnnLstmPolicy（这次环境才真的要求反应，六月那次 LSTM 实验不算数）
# 课表：先无抖动 4M 学会游泳/走完全程（不然从零直接上抖动大概什么都学不到），
#       再课程式加压 0-4 → 0-8 → 0-16 → 0-30 各 2M。合计 12M。
set -eux
cd "$(dirname "$0")"
ARM=$1; GPU=$2
case $ARM in
  ff4)  ARCH=ff;   STACK=4 ;;
  ff8)  ARCH=ff;   STACK=8 ;;
  lstm) ARCH=lstm; STACK=4 ;;
esac
BASE=__from_scratch__
PHASES="0:4000000 4:2000000 8:2000000 16:2000000 30:2000000"
for ph in $PHASES; do
  K=${ph%%:*}; STEPS=${ph##*:}
  OUT=mario_22form_${ARM}_$K
  CUDA_VISIBLE_DEVICES=$GPU MARIO_DEVICE=cuda MARIO_ARCH=$ARCH MARIO_STACK=$STACK MARIO_NOOP=$K \
    MARIO_BASE=$BASE MARIO_BASE_VECN=vecnormalize_$OUT.pkl \
    MARIO_CKPT_DIR=./checkpoints_22form_${ARM}_$K MARIO_OUT=$OUT \
    venv/bin/python -u train_2_2_noop.py $STEPS 48
  BASE=$OUT.zip
done
echo FORM_${ARM}_DONE
