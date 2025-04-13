import gymnasium as gym
from gymnasium.envs.registration import register
from simple_grid_env import SimpleGridEnv

from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
import ray

# Register your custom env
register(
    id="SimpleGrid-v0",
    entry_point="simple_grid_env:SimpleGridEnv",
)

# Start Ray
ray.shutdown()
ray.init(ignore_reinit_error=True)

# Updated DQN config (new RLlib API - env_runners instead of rollouts)
config = (
    DQNConfig()
    .environment(env="SimpleGrid-v0")
    .env_runners(num_env_runners=1)
    .training(gamma=0.99, lr=0.0005)
    .resources(num_gpus=0)
    .framework("torch")
)

# Build and train the algorithm
algo = config.build()
results = []

for i in range(20):
    result = algo.train()
    results.append(result["episode_reward_mean"])
    print(f"Iteration {i + 1}: Mean reward = {result['episode_reward_mean']:.2f}")

# Save rewards
with open("dqn_rllib_rewards.txt", "w") as f:
    for r in results:
        f.write(f"{r}\n")

algo.stop()
ray.shutdown()

print("DQN training complete.")
