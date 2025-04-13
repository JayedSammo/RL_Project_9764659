import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SimpleGridEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(SimpleGridEnv, self).__init__()
        self.grid_size = 5
        self.render_mode = render_mode
        self.observation_space = spaces.MultiDiscrete([self.grid_size, self.grid_size])
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right
        self.agent_pos = [0, 0]
        self.goal_pos = [4, 4]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = [0, 0]
        return np.array(self.agent_pos, dtype=np.int32), {}

    def step(self, action):
        x, y = self.agent_pos

        if action == 0 and y > 0:     # Up
            y -= 1
        elif action == 1 and y < self.grid_size - 1:  # Down
            y += 1
        elif action == 2 and x > 0:   # Left
            x -= 1
        elif action == 3 and x < self.grid_size - 1:  # Right
            x += 1

        self.agent_pos = [x, y]

        done = self.agent_pos == self.goal_pos
        reward = 1.0 if done else -0.1
        return np.array(self.agent_pos, dtype=np.int32), reward, done, False, {}

    def render(self):
        grid = np.full((self.grid_size, self.grid_size), ' . ')
        x, y = self.agent_pos
        gx, gy = self.goal_pos
        grid[gy][gx] = ' G '
        grid[y][x] = ' A '
        for row in grid:
            print(''.join(row))
        print()
