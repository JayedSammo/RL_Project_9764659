import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridEnv(gym.Env):
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.observation_space = spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # 0=up, 1=down, 2=left, 3=right
        self.reset()

    def reset(self, seed=None, options=None):
        self.agent_pos = np.array([0, 0], dtype=np.float32)
        self.goal_pos = np.array([self.grid_size - 1, self.grid_size - 1], dtype=np.float32)
        return self.agent_pos.astype(np.float32), {}

    def step(self, action):
        if action == 0 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.grid_size - 1:
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.grid_size - 1:
            self.agent_pos[1] += 1

        done = np.array_equal(self.agent_pos, self.goal_pos)
        reward = 1 if done else -0.1
        return self.agent_pos.astype(np.float32), reward, done, False, {}

    def render(self):
        grid = np.full((self.grid_size, self.grid_size), '.')
        grid[self.goal_pos[0], self.goal_pos[1]] = 'G'
        grid[self.agent_pos[0], self.agent_pos[1]] = 'A'
        print("\n".join([" ".join(row) for row in grid]))
        print()
