import ray
from ray import tune
from ray.tune import RunConfig
from ray.rllib.algorithms.dqn import DQNConfig
from custom_env.grid_env import GridEnv

# Register the custom environment
tune.register_env("CustomGrid-v0", lambda cfg: GridEnv(**cfg))

# Initialize Ray
ray.init()

# Define DQN configuration using the new API
config = (
    DQNConfig()
    .environment("CustomGrid-v0", env_config={"grid_size": 5})
    .env_runners(num_env_runners=1)  # replaces deprecated rollouts
    .training(train_batch_size=64)
    .framework("torch")
)

# Launch training with Tune
tune.Tuner(
    "DQN",
    param_space=config,
    run_config=RunConfig(
        name="DQN_CustomGrid",
        stop={"episode_reward_mean": 0.9}
    ),
).fit()
