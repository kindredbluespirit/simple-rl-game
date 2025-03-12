import torch
import torch.nn as nn
import gymnasium as gym
from il_train import ImitationAgent

# Load trained model
imitation_model = ImitationAgent()
imitation_model.load_state_dict(torch.load("imitation_model.pth"))
imitation_model.eval()

# Test in environment
env = gym.make("CarRacing-v3", render_mode="human")
obs, _ = env.reset()

for _ in range(1000):
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    obs_tensor = obs_tensor.permute(0, 3, 1, 2)
    # print(obs_tensor.shape)
    action = imitation_model(obs_tensor).detach().numpy()[0]  # Get predicted action
    obs, reward, done, truncated, _ = env.step(action)
    
    if done or truncated:
        obs, _ = env.reset()

env.close()
