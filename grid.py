import numpy as np
import pygame

GRID_SIZE = 20
CELL_SIZE = 500 // GRID_SIZE # Cell size in pixels
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
render_mode = False
WALLS = [
    [2,2], [2,3], [2,4],
    [5,5], [5,6], [5,7],
    [6,7], [7,7], [8,7],
    [10,10], [10,11], [10,12]
]

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
        self.walls = WALLS
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

        self.player[0] = np.clip(self.player[0] + dx, 0, self.size-1)
        self.player[1] = np.clip(self.player[1] + dy, 0, self.size-1)

        reward = -1
        done = False

        if self.player == self.goal:
            reward = 10
            done = True

        #self.render()
        return self.get_state(), reward, done
    
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
