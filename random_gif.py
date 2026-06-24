import warnings; warnings.filterwarnings("ignore")
import imageio, numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode='rgb_array')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
reset_out = env.reset()
obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

frames = [np.array(obs)]
for i in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    frames.append(np.array(obs))
    if terminated or truncated:
        ro = env.reset(); obs = ro[0] if isinstance(ro, tuple) else ro
env.close()

# sanity: how much do frames actually vary?
diffs = [np.abs(frames[i].astype(int)-frames[i-1].astype(int)).mean() for i in range(1,len(frames))]
print("mean frame-to-frame diff:", round(float(np.mean(diffs)),3), "| frame shape:", frames[0].shape)

frames = frames[::3]   # thin for size
imageio.mimsave('mario_random.gif', frames, fps=20)
print("saved mario_random.gif frames:", len(frames))
