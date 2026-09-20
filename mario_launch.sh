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
HOST=${MARIO_HOST:-158}          # 换机器时传 MARIO_HOST，别再写死
GPU=$1; LOG=$2; shift 2
ENVS=(); while [ "$1" != "--" ]; do ENVS+=("$1"); shift; done; shift

CUDA_VISIBLE_DEVICES=$GPU env "${ENVS[@]}" \
  setsid nohup ./venv/bin/python train_world_noop.py "$@" > $LOG.log 2>&1 < /dev/null &
PID=$!
echo "$LOG PID=$PID"

# 把本次 PID 记到共享清单里，收工时按清单判断"还有没有别的 mario 任务"
PIDFILE=/tmp/mario_pids
echo $PID >> $PIDFILE

setsid nohup bash -c "
  while kill -0 $PID 2>/dev/null; do
    gpuwatch heartbeat $HOST --owner xzh-claude >/dev/null 2>&1
    sleep 2400
  done
  sleep 30
  # ⚠️ 判'还有没有别的任务'**别用模式匹配**，两种错法都踩过：
  #   pgrep -f '…train_world_noop…' → 匹配到这段脚本自己的命令行，永远认为有任务，永远不释放
  #                                    （挂了一条 43 小时的 STALE 租约）
  #   pgrep -c -x python            → 共享机器上别人的 python 也算进来（实测某节点 276 个），
  #                                    同样永远不释放
  # 只认自己登记过的 PID：逐个 kill -0，全死光才释放。
  alive=0
  while read -r p; do kill -0 \"\$p\" 2>/dev/null && alive=\$((alive+1)); done < $PIDFILE
  if [ \"\$alive\" -eq 0 ]; then
    gpuwatch release $HOST --owner xzh-claude >/dev/null 2>&1
    : > $PIDFILE
  fi
" > /dev/null 2>&1 < /dev/null &
