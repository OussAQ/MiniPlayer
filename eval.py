import torch
import torch.nn as nn
import random
from grid import GridGame
from dqn import DQN

Niter = 10
render = True
env = GridGame(render_mode=render)
model = DQN()
model.load_state_dict(torch.load("model/dqn_model_episode_400.pth"))

for episode in range(Niter):
    state = env.reset()
    done = False

    while not done:
        if render:
            env.render()

        with torch.no_grad():
            action = torch.argmax(model(torch.tensor(state, dtype=torch.float32))).item()

        next_state, reward, done = env.step(action)

        state = next_state