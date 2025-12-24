"""Algorithm configuration for IMPALA training.

Usage:
    from training.config import get_algorithm_settings, create_algorithm_config
    
    settings = get_algorithm_settings()
    config = create_algorithm_config()
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from ray.rllib.algorithms.impala import ImpalaConfig


class AlgorithmSettings(BaseSettings):
    """Algorithm hyperparameters. All fields can be overridden via environment variables."""
    
    environment: str = Field(
        default="CartPole-v1",
        alias="ENVIRONMENT",
        description="Gym environment name"
    )
    num_rollout_workers: int = Field(
        default=4,
        alias="NUM_ROLLOUT_WORKERS",
        description="Number of parallel rollout workers"
    )
    gamma: float = Field(
        default=0.995,
        alias="GAMMA",
        description="Discount factor"
    )
    learning_rate: float = Field(
        default=5e-4,
        alias="LEARNING_RATE",
        description="Learning rate"
    )
    train_batch_size: int = Field(
        default=4096,
        alias="TRAIN_BATCH_SIZE",
        description="Training batch size"
    )
    entropy_coeff: float = Field(
        default=0.005,
        alias="ENTROPY_COEFF",
        description="Entropy coefficient for exploration"
    )
    vf_loss_coeff: float = Field(
        default=0.5,
        alias="VF_LOSS_COEFF",
        description="Value function loss coefficient"
    )
    grad_clip: float = Field(
        default=40.0,
        alias="GRAD_CLIP",
        description="Gradient clipping threshold"
    )
    min_time_s_per_iteration: int = Field(
        default=2,
        alias="MIN_TIME_S_PER_ITERATION",
        description="Minimum seconds per training iteration"
    )
    min_sample_timesteps_per_iteration: int = Field(
        default=4000,
        alias="MIN_SAMPLE_TIMESTEPS_PER_ITERATION",
        description="Minimum timesteps sampled per iteration"
    )
    num_gpus: int = Field(
        default=0,
        alias="NUM_GPUS",
        description="Number of GPUs to use"
    )
    seed: int = Field(
        default=42,
        alias="SEED",
        description="Random seed for reproducibility"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
        populate_by_name=True,
    )


class TrainingSettings(BaseSettings):
    """Training run settings. All fields can be overridden via environment variables."""
    
    model_name: str = Field(
        default="impala_cartpole",
        alias="MODEL_NAME",
        description="Name of the model being trained"
    )
    num_iterations: int = Field(
        default=100,
        alias="NUM_ITERATIONS",
        description="Number of training iterations"
    )
    checkpoint_interval: int = Field(
        default=10,
        alias="CHECKPOINT_INTERVAL",
        description="Save checkpoint every N iterations"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
        populate_by_name=True,
    )


class EvaluationSettings(BaseSettings):
    """Evaluation settings. All fields can be overridden via environment variables."""
    
    num_episodes: int = Field(
        default=100,
        alias="NUM_EVAL_EPISODES",
        description="Number of episodes to evaluate"
    )
    render: bool = Field(
        default=False,
        alias="RENDER",
        description="Whether to render the environment"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache(maxsize=1)
def get_algorithm_settings() -> AlgorithmSettings:
    """Get cached algorithm settings."""
    return AlgorithmSettings()


@lru_cache(maxsize=1)
def get_training_settings() -> TrainingSettings:
    """Get cached training settings."""
    return TrainingSettings()


@lru_cache(maxsize=1)
def get_evaluation_settings() -> EvaluationSettings:
    """Get cached evaluation settings."""
    return EvaluationSettings()


def create_algorithm_config(settings: AlgorithmSettings | None = None) -> ImpalaConfig:
    """Create and configure the IMPALA algorithm.
    
    Args:
        settings: Algorithm settings (uses cached defaults if not provided)
        
    Returns:
        Configured ImpalaConfig instance
    """
    if settings is None:
        settings = get_algorithm_settings()
    
    return (
        ImpalaConfig()
        .environment(settings.environment)
        .rollouts(num_rollout_workers=settings.num_rollout_workers)
        .training(
            gamma=settings.gamma,
            lr=settings.learning_rate,
            train_batch_size=settings.train_batch_size,
            entropy_coeff=settings.entropy_coeff,
            vf_loss_coeff=settings.vf_loss_coeff,
            grad_clip=settings.grad_clip,
        )
        .reporting(
            min_time_s_per_iteration=settings.min_time_s_per_iteration,
            min_sample_timesteps_per_iteration=settings.min_sample_timesteps_per_iteration,
        )
        .resources(num_gpus=settings.num_gpus)
        .debugging(log_level="ERROR", seed=settings.seed)
    )


if __name__ == "__main__":
    pass
