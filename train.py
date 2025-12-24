import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
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
    
    # Early termination tracking
    max_steps = 2 * env.size * env.size  # Maximum steps: 2 * grid area
    episode_steps = 0
    recent_positions = deque(maxlen=6)  # Track recent positions
    recent_actions = deque(maxlen=6)  # Track recent actions
    early_termination_reason = None

    while not done:
        episode_steps += 1
        
        # Early termination: too many steps
        if episode_steps > max_steps:
            done = True
            early_termination_reason = f"max_steps ({max_steps})"
            break
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                action = torch.argmax(model(torch.tensor(state, dtype=torch.float32))).item()

        next_state, reward, done = env.step(action)
        
        # Track position and action for cycle detection
        current_pos = tuple(env.player)
        recent_positions.append(current_pos)
        recent_actions.append(action)
        
        # Detect and penalize back-and-forth behavior
        cycle_penalty = 0.0
        if len(recent_actions) >= 4:
            last_4_actions = list(recent_actions)[-4:]
            
            # Pattern 1: A-B-A-B (back and forth)
            # Opposite pairs: (0,1)=up-down, (1,0)=down-up, (2,3)=left-right, (3,2)=right-left
            opposite_pairs = [(0, 1), (1, 0), (2, 3), (3, 2)]
            if (last_4_actions[0] == last_4_actions[2] and 
                last_4_actions[1] == last_4_actions[3] and
                (last_4_actions[0], last_4_actions[1]) in opposite_pairs):
                cycle_penalty = -2.0  # Penalty for back-and-forth
                # If repeated multiple times, terminate early
                if len(recent_actions) >= 6:
                    last_6_actions = list(recent_actions)[-6:]
                    if (last_6_actions[0] == last_6_actions[2] == last_6_actions[4] and
                        last_6_actions[1] == last_6_actions[3] == last_6_actions[5] and
                        (last_6_actions[0], last_6_actions[1]) in opposite_pairs):
                        done = True
                        early_termination_reason = "back_and_forth_cycle"
                        break
            
            # Pattern 2: Circular pattern - visiting same positions repeatedly
            if len(recent_positions) >= 4:
                last_4_positions = list(recent_positions)[-4:]
                # If positions repeat (A -> B -> A -> B) and not just staying in place
                if (last_4_positions[0] == last_4_positions[2] and 
                    last_4_positions[1] == last_4_positions[3] and
                    last_4_positions[0] != last_4_positions[1]):
                    cycle_penalty = -1.5  # Penalty for circular movement
                    # If repeated 3 times (6 steps), terminate
                    if len(recent_positions) >= 6:
                        last_6_positions = list(recent_positions)[-6:]
                        if (last_6_positions[0] == last_6_positions[2] == last_6_positions[4] and
                            last_6_positions[1] == last_6_positions[3] == last_6_positions[5] and
                            last_6_positions[0] != last_6_positions[1]):
                            done = True
                            early_termination_reason = "circular_cycle"
                            break
        
        # Apply cycle penalty to reward
        reward += cycle_penalty
        
        # Store transition with modified reward
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
            termination_info = f", Terminated: {early_termination_reason}" if early_termination_reason else ""
            print(f"Episode {episode+1}, Loss: {loss.item():.4f}, Epsilon: {epsilon:.2f}, Memory: {len(memory)}, Steps: {episode_steps}{termination_info}")
        else:
            print(f"Episode {episode+1}, Collecting samples... ({len(memory)}/{min_memory_size}), Epsilon: {epsilon:.2f}")

    if (episode+1) % 100 == 0:
        torch.save(model.state_dict(), f"model/dqn_model_episode_{episode+1}.pth")