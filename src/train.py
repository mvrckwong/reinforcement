"""Main training script for IMPALA algorithm on CartPole-v1 environment."""

from dotenv import load_dotenv

from configs.paths import get_paths
from configs.logging import get_logging_manager, upload_run_logs, logger
from configs.run_context import RunContext
from training import CheckpointManager, Trainer, create_algorithm_config
from training.config import ENVIRONMENT


# Training hyperparameters
NUM_ITERATIONS = 100
CHECKPOINT_INTERVAL = 10
MODEL_NAME = "impala_cartpole"


def main() -> None:
    """Main training function."""
    # Load environment variables
    load_dotenv(get_paths().env_file)
    
    # Single source of truth for run identity
    context = RunContext(model_name=MODEL_NAME)
    
    # Setup logging and checkpoint management with shared context
    get_logging_manager().setup(context)
    checkpoint_manager = CheckpointManager.from_env(context=context)
    
    # Create and build algorithm
    logger.info(f"Training IMPALA on {ENVIRONMENT}...")
    config = create_algorithm_config()
    algo = config.build()
    
    # Create trainer and run training
    trainer = Trainer(
        algo=algo,
        checkpoint_manager=checkpoint_manager,
        num_iterations=NUM_ITERATIONS,
        checkpoint_interval=CHECKPOINT_INTERVAL,
    )
    
    trainer.train()
    logger.success("Training completed!")
    
    # Upload logs to S3
    upload_run_logs(context)


if __name__ == "__main__":
    main()
