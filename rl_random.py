import gymnasium as gym
import numpy as np

env = gym.make("CarRacing-v3", render_mode="human")
state, _ = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Random actions
    state, reward, done, truncated, _ = env.step(action)
    if done or truncated:
        state, _ = env.reset()

env.close()

