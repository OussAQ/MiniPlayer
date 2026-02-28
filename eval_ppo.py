import torch
import torch.nn as nn
import random
from grid import GridGame
from ppo import PPO

Niter = 10
render = True
env = GridGame(render_mode=render)
# Calculate state size: 4 (player+goal positions) + grid_size^2 (wall grid)
state_size = 4 + env.size * env.size
action_size = 4  # up, down, left, right
model = PPO(state_size=state_size, action_size=action_size)
model.load_state_dict(torch.load("model/ppo_model_final.pth"))
seed = 42
random.seed(seed)
for episode in range(Niter):
    state = env.reset()
    done = False

    while not done:
        if render:
            env.render()

        with torch.no_grad():
            action_probs, _ = model(torch.tensor(state, dtype=torch.float32))
            action = torch.argmax(action_probs).item()

        next_state, reward, done = env.step(action)

        state = next_state