import ray
from ray import tune
from ray.tune import RunConfig
from ray.tune.callback import Callback
from ray.rllib.algorithms.dqn import DQNConfig
from custom_env.grid_env import GridEnv
import os

# Register custom environment
tune.register_env("CustomGrid-v0", lambda cfg: GridEnv(**cfg))

# Initialize Ray
ray.init()

# DQN config with compatible keys
config = (
    DQNConfig()
    .environment("CustomGrid-v0", env_config={"grid_size": 5})
    .framework("torch")
    .training(
        lr=0.0005,
        train_batch_size=64,
    )
    .exploration(explore=True)
)

# Reward logger
class RewardLogger(Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        rewards = result.get("episode_reward_mean", None)
        if rewards is not None:
            with open("dqn_rewards.txt", "a") as f:
                f.write(f"{rewards}\n")

# Run DQN training
tuner = tune.Tuner(
    "DQN",
    param_space=config,
    run_config=RunConfig(
        name="DQN_CustomGrid",
        stop={"training_iteration": 50},
        callbacks=[RewardLogger()],
    ),
)

# Launch training
tuner.fit()
print("DQN training finished.")
