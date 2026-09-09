#!/bin/bash
# 起训练 + 自动续租 + 收工自动释放。
# 用法: mario_launch.sh <gpu> <日志名> <env赋值...> -- <脚本参数...>
#
# ⚠️ 这个脚本是被两次事故逼出来的，两次的病因不同，都要治：
#  ① 一次性给足 TTL 然后不管 → 租约过期挂了 33 小时，八张卡在账本里空占。
#     治法：心跳，间隔远小于 TTL 一半，且跟训练进程绑生死。
#  ② 只有心跳还不够 → 训练结束心跳自停，但**租约不会自己消失**，
#     会以 STALE 状态继续挂在账本里（第二次挂了 17 小时，又得手工释放）。
#     治法：心跳循环退出时，确认机器上没有别的 mario 任务了，就自动释放。
set -e
cd /mnt/nfs/xzh/mario-rl
GPU=$1; LOG=$2; shift 2
ENVS=(); while [ "$1" != "--" ]; do ENVS+=("$1"); shift; done; shift

CUDA_VISIBLE_DEVICES=$GPU env "${ENVS[@]}" \
  setsid nohup ./venv/bin/python train_world_noop.py "$@" > $LOG.log 2>&1 < /dev/null &
PID=$!
echo "$LOG PID=$PID"

setsid nohup bash -c "
  while kill -0 $PID 2>/dev/null; do
    gpuwatch heartbeat 158 --owner xzh-claude >/dev/null 2>&1
    sleep 2400
  done
  # 自己这一份跑完了。还有别的 mario 任务在跑就别动租约（多个任务共用一份租约）；
  # 一个都没有了才释放，免得留一条 STALE 挂在账本上占着八张卡。
  sleep 30
  if ! pgrep -f 'train_world_noop|distill_all12|collect_distill|collect_dagger' >/dev/null 2>&1; then
    gpuwatch release 158 --owner xzh-claude >/dev/null 2>&1
  fi
" > /dev/null 2>&1 < /dev/null &
