from gymnasium.envs.registration import register
from custom_env.grid_env import GridEnv

register(
    id="CustomGrid-v0",
    entry_point="custom_env.grid_env:GridEnv",
)
