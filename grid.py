import numpy as np

class GridGame:
    def __init__(self, size=5):
        self.size = size    # Size of the grid
        self.reset()    # Initialize the game

    def reset(self):
        self.player = [0, 0]    # Starting position
        self.goal = [self.size - 1, self.size - 1]  # Goal position
        return self.get_state()

    def get_state(self):
        return np.array(self.player + self.goal, dtype=np.float32)

    def step(self, action):
        # 0=up, 1=down, 2=left, 3=right
        moves = [(-1,0), (1,0), (0,-1), (0,1)]
        dx, dy = moves[action]

        self.player[0] = np.clip(self.player[0] + dx, 0, self.size-1)
        self.player[1] = np.clip(self.player[1] + dy, 0, self.size-1)

        reward = -1
        done = False

        if self.player == self.goal:
            reward = 10
            done = True

        return self.get_state(), reward, done
