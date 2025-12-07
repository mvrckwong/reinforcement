from ray.rllib.algorithms.impala import ImpalaConfig
from tqdm import tqdm
from dotenv import load_dotenv
from typing import Any, Mapping
from datetime import datetime

from configs.paths import Paths

load_dotenv(Paths.ENV_FILE)

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

LR_START = 5e-4
LR_END = 1e-4
LR_SCHEDULE_TIMESTEPS = 100_000

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Configure IMPALA
config = (
    ImpalaConfig()
    .environment("CartPole-v1")
    .rollouts(num_rollout_workers=4)
    .training(
        gamma=0.995,
        lr=5e-4,
        train_batch_size=4096,
        entropy_coeff=0.005,
        vf_loss_coeff=0.5,
        grad_clip=40.0,
    )
    .reporting(
        # Wait for meaningful work each iteration
        min_time_s_per_iteration=2,
        min_sample_timesteps_per_iteration=4000,
    )
    .resources(num_gpus=0)
    .debugging(log_level="ERROR", seed=42)
)

# Build algorithm
algo = config.build()
print("Training IMPALA on CartPole-v1...")

# Create checkpoint directory with timestamp
checkpoint_dir = Paths.CHECKPOINTS_DIR / "impala_cartpole" / datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
print(f"Checkpoints will be saved to: {checkpoint_dir}")

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
    steps = _metric(result, "num_env_steps_sampled", default=_metric(result, "env_steps_sampled", 0))

    tqdm.write(
        f"Iter {i+1:2d} | Episodes: {episodes:6.0f} | Reward: {reward:6.2f} | Length: {length:6.2f} | Steps: {steps:8.0f}"
    )
    
    # Save checkpoint every 10 iterations
    if (i + 1) % 10 == 0:
        checkpoint_path = algo.save(str(checkpoint_dir))
        tqdm.write(f"Checkpoint saved at: {checkpoint_path}")

# Save final checkpoint
final_checkpoint = algo.save(str(checkpoint_dir))
print(f"Final checkpoint saved at: {final_checkpoint}")

algo.stop()