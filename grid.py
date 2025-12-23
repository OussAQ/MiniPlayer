import numpy as np
import pygame
import random
from collections import deque

GRID_SIZE = 20
CELL_SIZE = 500 // GRID_SIZE # Cell size in pixels
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
render_mode = False
LOW_RATIO = 10
HIGH_RATIO = 30
# WALLS = [
#     [2,0], [2,3], [2,4],
#     [5,5], [5,6], [5,7],
#     [6,7], [7,7], [8,7],
#     [10,5], [11,5], [12,5],
#     [10,8], [10,9], [10,10], [10,11], [10,12],
#     [10,10], [10,11], [10,12],
#     [0,15], [1,15], [2,15], [3,15], [4,15], [5,15],
#     [15,2], [15,3], [15,4], [15,5]
# ]

class GridGame:
    def __init__(self, size=GRID_SIZE, render_mode=render_mode):
        self.render_mode = render_mode
        if render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            self.caption = pygame.display.set_caption("AI Grid Game")
            self.clock = pygame.time.Clock()

        self.size = size    # Size of the grid
        self.reset()    # Initialize the game

    def reset(self):
        self.player = [0, 0]    # Starting position
        self.goal = [self.size - 1, self.size - 1]  # Goal position
        self.gen_ratio = random.randint(LOW_RATIO, HIGH_RATIO) / 100.0
        self.generate_walls(self.gen_ratio)
        return self.get_state()

    def get_state(self):
        return np.array(self.player + self.goal, dtype=np.float32)

    def step(self, action): # 0=up, 1=down, 2=left, 3=right
        if self.render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()
            
        moves = [(-1,0), (1,0), (0,-1), (0,1)]
        dx, dy = moves[action]
        # Move player if no wall
        if not self.hit_wall(dx, dy):
            self.player[0] = np.clip(self.player[0] + dx, 0, self.size-1)
            self.player[1] = np.clip(self.player[1] + dy, 0, self.size-1)
            reward = -1
        else:
            reward = -5
        done = False

        if self.player == self.goal:
            reward = 10
            done = True

        #self.render()
        return self.get_state(), reward, done

    def hit_wall(self, dx, dy):
        return [self.player[0] + dx, self.player[1] + dy] in self.walls
    
    def is_reachable(self):
        # Simple BFS to ensure the goal is reachable from the start
        queue = deque()
        queue.append(self.player)
        visited = set()
        visited.add(tuple(self.player))

        while queue:
            x, y = queue.popleft()
            if [x, y] == self.goal:
                return True
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size and [nx, ny] not in self.walls and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append([nx, ny])
        return False
    
    def generate_walls(self, gen_ratio=0.2):
        self.walls = []
        while True:
            self.walls = []
            for x in range(self.size):
                for y in range(self.size):
                    if random.random() < gen_ratio and [x, y] != self.player and [x, y] != self.goal:
                        self.walls.append([x, y])
            if self.is_reachable():
                break

    def render(self):
        self.screen.fill((30, 30, 30))
        # Draw grid
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(
                    y * CELL_SIZE,
                    x * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE
                )
                pygame.draw.rect(self.screen, (60, 60, 60), rect, 1)

        # Goal (green)
        gx, gy = self.goal
        pygame.draw.rect(
            self.screen,
            (0, 200, 0),
            pygame.Rect(
                gy * CELL_SIZE,
                gx * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )
        )

        # Player (blue)
        px, py = self.player
        pygame.draw.rect(
            self.screen,
            (50, 150, 255),
            pygame.Rect(
                py * CELL_SIZE,
                px * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )
        )

        # Walls (gray)
        for wall in self.walls:
            wx, wy = wall
            pygame.draw.rect(
                self.screen,
                (100, 100, 100),
                pygame.Rect(
                    wy * CELL_SIZE,
                    wx * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE
                )
            )

        pygame.display.flip()
        self.clock.tick(10)
