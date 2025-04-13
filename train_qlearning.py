import gymnasium as gym
import numpy as np
import random
import custom_env  # triggers registration of CustomGrid-v0

alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
episodes = 500

env = gym.make("CustomGrid-v0", grid_size=5)
q_table = np.zeros((5, 5, env.action_space.n))
reward_log = []

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        x, y = state
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[x, y])

        next_state, reward, done, _, _ = env.step(action)
        nx, ny = next_state
        old_value = q_table[x, y, action]
        next_max = np.max(q_table[nx, ny])
        q_table[x, y, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        state = next_state
        total_reward += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    reward_log.append(total_reward)

    if (episode + 1) % 50 == 0:
        print(f"Episode {episode + 1}, Total reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

env.close()

with open("q_learning_rewards.txt", "w") as f:
    for r in reward_log:
        f.write(f"{r}\n")

print("Training complete.")
