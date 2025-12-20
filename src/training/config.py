"""Algorithm configuration module."""

from ray.rllib.algorithms.impala import ImpalaConfig


# Configuration constants
ENVIRONMENT = "CartPole-v1"
NUM_ROLLOUT_WORKERS = 4
GAMMA = 0.995
LEARNING_RATE = 5e-4
TRAIN_BATCH_SIZE = 4096
ENTROPY_COEFF = 0.005
VF_LOSS_COEFF = 0.5
GRAD_CLIP = 40.0
MIN_TIME_S_PER_ITERATION = 2
MIN_SAMPLE_TIMESTEPS_PER_ITERATION = 4000
NUM_GPUS = 0
SEED = 42


def create_algorithm_config(
    environment: str = ENVIRONMENT,
    num_workers: int = NUM_ROLLOUT_WORKERS,
    gamma: float = GAMMA,
    lr: float = LEARNING_RATE,
    train_batch_size: int = TRAIN_BATCH_SIZE,
    entropy_coeff: float = ENTROPY_COEFF,
    vf_loss_coeff: float = VF_LOSS_COEFF,
    grad_clip: float = GRAD_CLIP,
    num_gpus: int = NUM_GPUS,
    seed: int = SEED,
) -> ImpalaConfig:
    """Create and configure the IMPALA algorithm.
    
    Args:
        environment: Gym environment name
        num_workers: Number of rollout workers
        gamma: Discount factor
        lr: Learning rate
        train_batch_size: Training batch size
        entropy_coeff: Entropy coefficient
        vf_loss_coeff: Value function loss coefficient
        grad_clip: Gradient clipping threshold
        num_gpus: Number of GPUs to use
        seed: Random seed for reproducibility
        
    Returns:
        Configured ImpalaConfig instance
    """
    return (
        ImpalaConfig()
        .environment(environment)
        .rollouts(num_rollout_workers=num_workers)
        .training(
            gamma=gamma,
            lr=lr,
            train_batch_size=train_batch_size,
            entropy_coeff=entropy_coeff,
            vf_loss_coeff=vf_loss_coeff,
            grad_clip=grad_clip,
        )
        .reporting(
            min_time_s_per_iteration=MIN_TIME_S_PER_ITERATION,
            min_sample_timesteps_per_iteration=MIN_SAMPLE_TIMESTEPS_PER_ITERATION,
        )
        .resources(num_gpus=num_gpus)
        .debugging(log_level="ERROR", seed=seed)
    )


if __name__ == "__main__":
    pass