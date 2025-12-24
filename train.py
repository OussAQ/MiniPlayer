import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dqn import DQN, ReplayMemory
from grid import GridGame

env = GridGame()
# Calculate state size: 4 (player+goal positions) + grid_size^2 (wall grid)
state_size = 4 + env.size * env.size
model = DQN(state_size=state_size)

# Target network for stable Q-learning
target_model = DQN(state_size=state_size)  # Target network for stable Q-learning
target_model.load_state_dict(model.state_dict())  # Initialize with same weights
target_model.eval()  # Set to evaluation mode

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Replay memory parameters
memory = ReplayMemory(10000)
batch_size = 64
min_memory_size = 2000  # Increased to avoid early bad samples
update_target_every = 10  # Update target network every N episodes
train_every = 4  # Train every N steps instead of every step

gamma = 0.99
epsilon = 1.0
render = env.render_mode
step_count = 0

for episode in range(500):
    state = env.reset()
    done = False

    while not done:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                action = torch.argmax(model(torch.tensor(state, dtype=torch.float32))).item()

        next_state, reward, done = env.step(action)
        memory.push(state, action, next_state, reward, done)
        step_count += 1

        # Train less frequently
        if len(memory) >= min_memory_size and step_count % train_every == 0:
            # Sample a batch from replay memory
            transitions = memory.sample(batch_size)
            batch = list(zip(*transitions))
            
            # Convert to numpy arrays first, then to tensors (fixes warning)
            state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float32)
            action_batch = torch.tensor(np.array(batch[1]), dtype=torch.long)
            next_state_batch = torch.tensor(np.array(batch[2]), dtype=torch.float32)
            reward_batch = torch.tensor(np.array(batch[3]), dtype=torch.float32)
            done_batch = torch.tensor(np.array(batch[4]), dtype=torch.bool)

            # Compute Q(s, a) for current states using main network
            state_action_values = model(state_batch).gather(1, action_batch.unsqueeze(1))

            # Compute Q(s', a') for next states using TARGET network (stable)
            with torch.no_grad():
                next_state_values = target_model(next_state_batch).max(1)[0].detach()
                expected_state_action_values = reward_batch + (gamma * next_state_values * ~done_batch)

            # Compute loss
            loss = loss_fn(state_action_values.squeeze(), expected_state_action_values)

            # Optimize with gradient clipping to prevent exploding gradients
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
            optimizer.step()

        state = next_state

    epsilon = max(0.1, epsilon * 0.995)

    # Update target network periodically
    if (episode + 1) % update_target_every == 0:
        target_model.load_state_dict(model.state_dict())
        print(f"Target network updated at episode {episode+1}")

    if (episode+1) % 10 == 0:
        if len(memory) >= min_memory_size:
            print(f"Episode {episode+1}, Loss: {loss.item():.4f}, Epsilon: {epsilon:.2f}, Memory: {len(memory)}")
        else:
            print(f"Episode {episode+1}, Collecting samples... ({len(memory)}/{min_memory_size}), Epsilon: {epsilon:.2f}")

    if (episode+1) % 100 == 0:
        torch.save(model.state_dict(), f"model/dqn_model_episode_{episode+1}.pth")