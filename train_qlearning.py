import gymnasium as gym
from gymnasium.envs.registration import register
from simple_grid_env import SimpleGridEnv
import numpy as np
import random

# Register the environment
register(
    id="SimpleGrid-v0",
    entry_point="simple_grid_env:SimpleGridEnv",
)

# Q-learning parameters
alpha = 0.1          # Learning rate
gamma = 0.99         # Discount factor
epsilon = 1.0        # Exploration rate
epsilon_min = 0.1
epsilon_decay = 0.995
episodes = 500

env = gym.make("SimpleGrid-v0")
q_table = np.zeros((5, 5, env.action_space.n))  # shape = [grid_x, grid_y, actions]

reward_log = []

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        x, y = state
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[x, y])   # Exploit

        next_state, reward, done, _, _ = env.step(action)
        nx, ny = next_state

        # Q-learning update
        old_value = q_table[x, y, action]
        next_max = np.max(q_table[nx, ny])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[x, y, action] = new_value

        state = next_state
        total_reward += reward

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    reward_log.append(total_reward)

    if (episode + 1) % 50 == 0:
        print(f"Episode {episode + 1}, Total reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

env.close()

# Save rewards for plotting later if needed
with open("q_learning_rewards.txt", "w") as f:
    for r in reward_log:
        f.write(f"{r}\n")

print("Training complete.")
