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

for episode in range(400):
    state = env.reset()
    done = False

    while not done:
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                action = torch.argmax(model(torch.tensor(state, dtype=torch.float32))).item()

        next_state, reward, done = env.step(action)

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

    if (episode+1) % 50 == 0:
        print(f"Episode {episode+1}, epsilon={epsilon:.2f}")