import ray
from ray import tune
from ray.tune import RunConfig
from ray.rllib.algorithms.ppo import PPOConfig
from custom_env.grid_env import GridEnv

# Register the custom environment
tune.register_env("CustomGrid-v0", lambda cfg: GridEnv(**cfg))

ray.init()

# Custom callback to collect episode rewards
from ray.rllib.algorithms.callbacks import DefaultCallbacks

class RewardLogger(DefaultCallbacks):
    def on_episode_end(self, *, episode, **kwargs):
        reward = episode.total_reward
        with open("ppo_rewards.txt", "a") as f:
            f.write(f"{reward}\n")

config = (
    PPOConfig()
    .environment("CustomGrid-v0", env_config={"grid_size": 5})
    .env_runners(num_env_runners=1)
    .training(train_batch_size=64)
    .framework("torch")
    .callbacks(RewardLogger)
)

tune.Tuner(
    "PPO",
    param_space=config,
    run_config=RunConfig(
        name="PPO_CustomGrid",
        stop={"training_iteration": 10}
    ),
).fit()
