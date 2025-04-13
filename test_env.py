import gymnasium as gym
import custom_env  # this triggers the registration

env = gym.make("CustomGrid-v0", grid_size=5)
obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render()
