from stable_baselines3 import PPO
import gymnasium as gym

model = PPO.load("car_racing_ppo")

env = gym.make("CarRacing-v3", render_mode="human")
obs, _ = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, _ = env.step(action)
    if done or truncated:
        obs, _ = env.reset()

env.close()

