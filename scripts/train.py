from ray.rllib.algorithms.impala import IMPALAConfig
from tqdm import tqdm
from dotenv import load_dotenv
from typing import Any, Mapping

load_dotenv()

# Dynamic LR schedule settings
LR_START = 5e-4
LR_END = 1e-4
# Approximate over 100 iterations x 1k timesteps/iter => 100k timesteps
LR_SCHEDULE_TIMESTEPS = 100_000

# Configure IMPALA
config = (
    IMPALAConfig()
    .environment("CartPole-v1")
    .learners(num_learners=0)
    .env_runners(num_env_runners=4)
    .training(
        gamma=0.99,
        lr=[
            [0, LR_START],
            [LR_SCHEDULE_TIMESTEPS, LR_END],
        ],
        train_batch_size=512,
    )
    .callbacks(
        # Use new callback-based loggers instead of deprecated ones
        callbacks_class=None  # Disable default callbacks
    )
    .reporting(
        # Wait for meaningful work each iteration
        min_time_s_per_iteration=1,
        min_sample_timesteps_per_iteration=1000,
    )
    .resources(num_gpus=0)
    .debugging(log_level="ERROR")
)

# Build algorithm
algo = config.build_algo()
print("Training IMPALA on CartPole-v1...")


def _metric(result: Mapping[str, Any], key: str, default: float | int = 0) -> float | int:
    """Return metric from result, checking env_runners first, then top-level.
    Falls back to default if neither exists.
    """
    if isinstance(result, dict):
        env = result.get("env_runners")
        if isinstance(env, dict) and key in env:
            return env.get(key, default)
        return result.get(key, default)
    return default

for i in tqdm(range(100), desc="Training", unit="iter"):
    result = algo.train()
    # Extract metrics with a small helper
    episodes = _metric(result, "num_episodes", default=_metric(result, "episodes_total", 0))
    reward = _metric(result, "episode_return_mean", default=_metric(result, "episode_reward_mean", 0.0))
    length = _metric(result, "episode_len_mean", default=0.0)

    tqdm.write(f"Iter {i+1:2d} | Episodes: {episodes:6.0f} | Reward: {reward:6.2f} | Length: {length:6.2f}")

algo.stop()