import random
import torch
import torch.nn as nn
import torch.optim as optim
from dqn import DQN
from grid import GridGame

env = GridGame()
model = DQN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

gamma = 0.99
epsilon = 1.0
render = env.render_mode

for episode in range(400):
    state = env.reset()
    done = False

    while not done:
        # Epsilon-greedy action selection
        if random.random() < epsilon:   # Prioritize exploration in the beginning
            action = random.randint(0, 3)
        else:
            with torch.no_grad():   # Exploit the learned policy
                action = torch.argmax(model(torch.tensor(state, dtype=torch.float32))).item()

        next_state, reward, done = env.step(action) # Take action in the environment

        with torch.no_grad():
            target = reward
            if not done:
                target += gamma * torch.max(
                    model(torch.tensor(next_state, dtype=torch.float32))
                ).item()

        prediction = model(torch.tensor(state, dtype=torch.float32))[action]
        loss = loss_fn(prediction, torch.tensor(target, dtype=torch.float32))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

    epsilon = max(0.1, epsilon * 0.995)

    if (episode+1) % 10 == 0:
        print(f"Episode {episode+1}, Loss: {loss.item():.4f}, Epsilon: {epsilon:.2f}")

    if (episode+1) % 100 == 0:
        torch.save(model.state_dict(), f"model/dqn_model_episode_{episode+1}.pth")