import gymnasium as gym
import numpy as np
import pickle
from stable_baselines3 import PPO

# Load trained RL model
model = PPO.load("car_racing_ppo")

# Setup environment
env = gym.make("CarRacing-v3")
obs, _ = env.reset()

# Storage for imitation learning dataset
dataset = []

for _ in range(5000):  # Collect 5000 steps of data
    action, _ = model.predict(obs)
    dataset.append((obs, action))
    obs, reward, done, truncated, _ = env.step(action)
    
    if done or truncated:
        obs, _ = env.reset()

env.close()

# Save dataset
with open("imitation_data.pkl", "wb") as f:
    pickle.dump(dataset, f)

print("Dataset saved successfully!")
