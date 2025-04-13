import ray
from ray import tune
from ray.tune import RunConfig
from ray.rllib.algorithms.pg import PGConfig
import custom_env  # This registers CustomGrid-v0

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Define the PG configuration
config = PGConfig().environment(
    env="CustomGrid-v0",
    env_config={"grid_size": 5}
)

# Run training using the latest Tuner API
tuner = tune.Tuner(
    "PG",
    param_space=config.to_dict(),
    run_config=RunConfig(
        stop={"episode_reward_mean": 0.9},
        name="PG_GridWorld"
    ),
)

results = tuner.fit()
