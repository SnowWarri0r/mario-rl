"""十二关成果 montage：单个网络依次打 1-1..1-4 / 2-1..2-4 / 3-1..3-4，各录一个通关局拼成一条 GIF。

跟八关版同套路，两处为 H20 改的：
① 12 关并行录（每关一个进程），Mac 上串行是十几分钟，这里 = 最慢那一关的时间；
② 帧不经进程间 pickle（12 关原始帧近 1GB），各自落盘 .npy，父进程再顺序拼。
用法: python record_all12_montage.py [model.zip] [out.gif]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, tempfile
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import imageio

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mario_all12_wide.zip"
OUT = sys.argv[2] if len(sys.argv) > 2 else "mario_all12.gif"
STAGES = [f"{w}-{s}" for w in (1, 2, 3) for s in (1, 2, 3, 4)]
TRIES = 60                                                  # 每关最多试 60 局找一个通关局
TMP = tempfile.mkdtemp(prefix="mario_montage_")


def record_stage(stage):
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import MarioBase, SkipFrame, GrayResize, FrameStack
    import wide_cnn  # noqa: F401  注册 WideNatureCNN 供反序列化

    model = PPO.load(MODEL, device="cpu")
    raw = MarioBase(stages=[stage])
    env = FrameStack(GrayResize(SkipFrame(raw, 4)), 4)
    start = (int(stage[0]), int(stage[2]))
    best, best_x, got = None, -1, False
    for _ in range(TRIES):
        o, _ = env.reset()
        frames, maxx, flag, w, s = [], 0, False, start[0], start[1]
        done = False
        while not done:
            a, _ = model.predict(o, deterministic=False)
            o, r, term, trunc, info = env.step(int(a)); done = term or trunc
            frames.append(np.array(raw.render()))          # 必须拷贝：nes-py 的屏幕是原地覆盖的同一块内存
            maxx = max(maxx, info.get("x_pos", 0)); flag = flag or info.get("flag_get", False)
            w, s = info.get("world", w), info.get("stage", s)
        if flag or (w, s) != start:
            best, got = frames, True
            break
        if maxx > best_x:
            best_x, best = maxx, frames
    env.close()
    path = os.path.join(TMP, f"{stage}.npy")
    np.save(path, np.asarray(best[::2], dtype=np.uint8))     # 隔帧抽稀，跟八关版一致
    return stage, got, best_x, len(best[::2]), path


def main():
    print(f"=== 录 {MODEL} 的十二关 montage（12 关并行，每关最多 {TRIES} 局）===", flush=True)
    with ProcessPoolExecutor(max_workers=len(STAGES)) as pool:
        results = list(pool.map(record_stage, STAGES))
    by_stage = {r[0]: r for r in results}
    total = 0
    with imageio.get_writer(OUT, mode="I", fps=20) as wr:
        for st in STAGES:                                   # 按关卡顺序拼，不按完成顺序
            _, got, best_x, n, path = by_stage[st]
            print(f"  {st}: {'通关✓' if got else f'最远x={best_x}'} ({n}帧)")
            for fr in np.load(path):
                wr.append_data(fr)
            total += n
            os.remove(path)
    os.rmdir(TMP)
    print(f">>> {OUT}  共 {total} 帧")


if __name__ == "__main__":
    main()
