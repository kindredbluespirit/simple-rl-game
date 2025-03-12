import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader, TensorDataset

# Load dataset
with open("imitation_data.pkl", "rb") as f:
    dataset = pickle.load(f)

states, actions = zip(*dataset)
states = np.array(states, dtype=np.float32).transpose(0, 3, 1, 2)  # Change shape to (N, C, H, W)
actions = np.array(actions, dtype=np.float32)

# Convert to tensors
states_tensor = torch.tensor(states)
actions_tensor = torch.tensor(actions)

# print(states_tensor.shape, actions_tensor.shapez)

# Create dataset & dataloader
train_dataset = TensorDataset(states_tensor, actions_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define CNN-based imitation learning model
class ImitationAgent(nn.Module):
    def __init__(self):
        super(ImitationAgent, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 10 * 10, 128),  # Adjust size based on the CNN output
            nn.ReLU(),
            nn.Linear(128, actions_tensor.shape[1])
        )

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)
        x = self.cnn(x)
        return self.fc(x)

def train():
    # Initialize model
    imitation_model = ImitationAgent()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(imitation_model.parameters(), lr=0.001)

    # Train model
    for epoch in range(10):  # Train for 10 epochs
        for batch_states, batch_actions in train_loader:
            optimizer.zero_grad()
            predicted_actions = imitation_model(batch_states)
            loss = criterion(predicted_actions, batch_actions)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss = {loss.item()}")

    # Save trained imitation model
    torch.save(imitation_model.state_dict(), "imitation_model.pth")
    print("Imitation model trained and saved!")

if __name__ == "__main__":
    train()