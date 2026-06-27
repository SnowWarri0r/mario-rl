import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode='rgb_array')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

reset_out = env.reset()
obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
print("reset OK, frame shape:", obs.shape)

total_r = 0
for i in range(50):
    action = env.action_space.sample()
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
    else:
        obs, reward, done, info = out
    total_r += reward
    if done:
        env.reset()
print("stepped 50 frames OK, total reward:", total_r)
print("mario x_pos:", info.get('x_pos'), "world:", info.get('world'), "stage:", info.get('stage'))
env.close()
print("ALL GOOD")
