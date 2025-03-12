from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("CarRacing-v3")
model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)  # Train for 100k steps

model.save("car_racing_ppo")
env.close()

