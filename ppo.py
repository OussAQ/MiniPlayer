import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPO(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PPO, self).__init__()
        # Shared feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        features = self.feature(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

    def get_action(self, state):
        """Sample action from policy"""
        with torch.no_grad():
            action_probs, _ = self.forward(state)
        action = torch.multinomial(action_probs, 1).item()
        return action, action_probs[0, action].item()