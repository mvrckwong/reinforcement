"""Metrics extraction and reporting utilities."""

from typing import Any, Mapping
from tqdm import tqdm


def extract_metric(
    result: Mapping[str, Any], 
    key: str, 
    default: float | int = 0
) -> float | int:
    """Extract metric from training result.
    
    Checks env_runners first, then top-level. Falls back to default if neither exists.
    
    Args:
        result: Training result dictionary
        key: Metric key to extract
        default: Default value if key not found
        
    Returns:
        Metric value or default
    """
    if isinstance(result, dict):
        env = result.get("env_runners")
        if isinstance(env, dict) and key in env:
            return env.get(key, default)
        return result.get(key, default)
    return default


def print_training_metrics(
    iteration: int,
    result: Mapping[str, Any],
) -> None:
    """Print formatted training metrics.
    
    Args:
        iteration: Current iteration number
        result: Training result dictionary
    """
    episodes = extract_metric(
        result, 
        "num_episodes", 
        default=extract_metric(result, "episodes_total", 0)
    )
    reward = extract_metric(
        result, 
        "episode_return_mean", 
        default=extract_metric(result, "episode_reward_mean", 0.0)
    )
    length = extract_metric(
        result, 
        "episode_len_mean", 
        default=0.0
    )
    steps = extract_metric(
        result, 
        "num_env_steps_sampled", 
        default=extract_metric(result, "env_steps_sampled", 0)
    )

    tqdm.write(
        f"Iter {iteration:2d} | "
        f"Episodes: {episodes:6.0f} | "
        f"Reward: {reward:6.2f} | "
        f"Length: {length:6.2f} | "
        f"Steps: {steps:8.0f}"
    )


if __name__ == "__main__":
    pass