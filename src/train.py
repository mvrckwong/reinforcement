"""Main training script for IMPALA algorithm on CartPole-v1 environment."""

from dotenv import load_dotenv
from loguru import logger

from configs.paths import get_paths
from configs.run_context import RunContext
from services.logs import get_logging_manager, upload_run_logs
from training import (
    CheckpointManager,
    Trainer,
    create_algorithm_config,
    get_algorithm_settings,
    get_training_settings,
)


def main() -> None:
    """Main training function."""
    # Load environment variables
    load_dotenv(get_paths().env_file)
    
    # Load settings from environment
    algo_settings = get_algorithm_settings()
    training_settings = get_training_settings()
    
    # Single source of truth for run identity
    context = RunContext(model_name=training_settings.model_name)
    
    # Setup logging and checkpoint management with shared context
    get_logging_manager().setup(context)
    checkpoint_manager = CheckpointManager.from_env(context=context)
    
    # Create and build algorithm
    logger.info(f"Training IMPALA on {algo_settings.environment}...")
    config = create_algorithm_config(algo_settings)
    algo = config.build()
    
    # Create trainer and run training
    trainer = Trainer(
        algo=algo,
        checkpoint_manager=checkpoint_manager,
        num_iterations=training_settings.num_iterations,
        checkpoint_interval=training_settings.checkpoint_interval,
    )
    
    trainer.train()
    logger.success("Training completed!")
    
    # Upload logs to S3
    upload_run_logs(context)


if __name__ == "__main__":
    main()
