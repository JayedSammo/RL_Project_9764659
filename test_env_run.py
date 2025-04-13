import gymnasium as gym
from gymnasium.envs.registration import register
from simple_grid_env import SimpleGridEnv
import random
import time

# Register the environment
register(
    id="SimpleGrid-v0",
    entry_point="simple_grid_env:SimpleGridEnv",
)

# Create the environment
env = gym.make("SimpleGrid-v0", render_mode=None)
obs, _ = env.reset()

done = False
total_reward = 0

print("Initial Observation:", obs)

while not done:
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    env.render()
    print(f"Action: {action}, Observation: {obs}, Reward: {reward}, Done: {done}")
    total_reward += reward
    time.sleep(0.3)  # slow down rendering

env.close()
print(f"Total reward: {total_reward}")
