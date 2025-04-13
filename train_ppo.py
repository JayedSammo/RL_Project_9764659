import ray
from ray import tune
from ray.tune import RunConfig
from ray.tune.callback import Callback
from ray.rllib.algorithms.ppo import PPOConfig
from custom_env.grid_env import GridEnv
import os

# Register custom environment
tune.register_env("CustomGrid-v0", lambda cfg: GridEnv(**cfg))

# Initialize Ray
ray.init()

# PPO config with compatible keys
config = (
    PPOConfig()
    .environment("CustomGrid-v0", env_config={"grid_size": 5})
    .env_runners(num_env_runners=1)
    .training(
        train_batch_size_per_learner=128,
        minibatch_size=128,
        num_sgd_iter=10,
        lr=0.0003,
    )
    .framework("torch")
)

# Reward logger
class RewardLogger(Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        rewards = result.get("episode_reward_mean", None)
        if rewards is not None:
            with open("ppo_rewards.txt", "a") as f:
                f.write(f"{rewards}\n")

# Run PPO training
tuner = tune.Tuner(
    "PPO",
    param_space=config,
    run_config=RunConfig(
        name="PPO_CustomGrid",
        stop={"training_iteration": 50},
        callbacks=[RewardLogger()],
    ),
)

tuner.fit()
print("PPO training finished.")
