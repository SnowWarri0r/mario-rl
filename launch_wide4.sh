#!/bin/bash
# 宽窗口铺开：按鲁棒性余量排的四关。每条独立 setsid，避免串联启动时后面的被父 shell 带走
cd /mnt/nfs/xzh/mario-rl
launch() {  # $1=gpu $2=stage $3=out $4=base $5=pkl $6=steps
  CUDA_VISIBLE_DEVICES=$1 setsid nohup env MARIO_NOOP=120 MARIO_STAGE=$2 MARIO_DEVICE=cuda \
    MARIO_OUT=$3 MARIO_BASE=$4 MARIO_BASE_VECN=$5 \
    MARIO_ENT=0 MARIO_LR=1e-5 MARIO_SAVE_FREQ=250000 \
    venv/bin/python -u train_world_noop.py single $6 64 > train_$3.log 2>&1 < /dev/null &
  echo "  已起 $2 -> $3 (GPU $1)"
}
launch 0 2-3 mario_23wide mario_w2land_final.zip vecnormalize_w2land.pkl 6000000
launch 1 1-4 mario_14wide checkpoints_w1ent0/w1ent0_3500000_steps.zip vecnormalize_w1ent0.pkl 6000000
launch 2 1-1 mario_11wide checkpoints_w1ent0/w1ent0_3500000_steps.zip vecnormalize_w1ent0.pkl 6000000
launch 3 2-2 mario_22wided mario_22robust.zip vecnormalize_mario_22wide120b.pkl 8000000
sleep 5; echo "存活: $(ps -eo args | grep -c '[t]rain_world_noop.py')"
