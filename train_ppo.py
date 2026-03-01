import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from grid import GridGame
from ppo import PPO

# Initialize environment and device
env = GridGame()
state_size = 4 + env.size * env.size
action_size = 4  # up, down, left, right
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LEARNING_RATE = 3e-4
GAMMA = 0.99 
GAE_LAMBDA = 0.95 
EPOCHS = 10
BATCH_SIZE = 64
CLIP_RATIO = 0.2 
ENTROPY_COEFF = 0.05  # Entropy bonus (increased for better exploration)
model = PPO(state_size=state_size, action_size=action_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95):
    """Compute Generalized Advantage Estimation"""
    advantages = []
    gae = 0
    next_value = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lambda_ * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    return np.array(advantages)


def train_ppo(num_episodes=1000, update_freq=1024, max_steps_per_episode=100):
    episode_rewards = []
    step_count = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        episode_steps = 0
        
        done = False
        while not done and episode_steps < max_steps_per_episode:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Get action from policy
            with torch.no_grad():
                action_probs, value = model(state_tensor)
            
            action = torch.multinomial(action_probs, 1).item()
            log_prob = torch.log(action_probs[0, action]).item()
            value = value.item()
            
            # Step environment
            next_state, reward, done = env.step(action)
            # Reward shaping now handled in grid.py
            
            # Store trajectory
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            dones.append(done)
            
            episode_reward += reward
            state = next_state
            episode_steps += 1
            step_count += 1
            
            # Update when trajectory is long enough or episode done
            if step_count % update_freq == 0 or done or episode_steps >= max_steps_per_episode:
                # Compute advantages
                advantages = compute_gae(rewards, values, dones, gamma=GAMMA, lambda_=GAE_LAMBDA)
                returns = advantages + np.array(values)
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Convert to tensors
                states_tensor = torch.FloatTensor(np.array(states)).to(device)
                actions_tensor = torch.LongTensor(actions).to(device)
                log_probs_tensor = torch.FloatTensor(log_probs).to(device)
                advantages_tensor = torch.FloatTensor(advantages).to(device)
                returns_tensor = torch.FloatTensor(returns).to(device)
                
                # PPO update
                for epoch in range(EPOCHS):
                    # Shuffle batch
                    indices = np.random.permutation(len(states))
                    
                    for i in range(0, len(states), BATCH_SIZE):
                        batch_indices = indices[i:i+BATCH_SIZE]
                        
                        action_probs, values_pred = model(states_tensor[batch_indices])
                        new_log_probs = torch.log(action_probs.gather(1, actions_tensor[batch_indices].unsqueeze(1))).squeeze(1)
                        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(1).mean()
                        
                        # PPO loss
                        ratio = torch.exp(new_log_probs - log_probs_tensor[batch_indices])
                        surr1 = ratio * advantages_tensor[batch_indices]
                        surr2 = torch.clamp(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO) * advantages_tensor[batch_indices]
                        actor_loss = -torch.min(surr1, surr2).mean()
                        
                        # Value loss
                        critic_loss = nn.MSELoss()(values_pred.squeeze(1), returns_tensor[batch_indices])
                        
                        # Total loss
                        loss = actor_loss + critic_loss - ENTROPY_COEFF * entropy
                        
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        optimizer.step()
                
                # Reset trajectory
                states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward (last 50): {avg_reward:.2f}")
        
        if (episode + 1) % 200 == 0:
            torch.save(model.state_dict(), f"model/ppo_model_episode_{episode + 1}.pth")
            print(f"Model saved at episode {episode + 1}")
    
    return episode_rewards


if __name__ == "__main__":
    rewards = train_ppo(num_episodes=10000, update_freq=1024)
    torch.save(model.state_dict(), "model/ppo_model_final.pth")