"""Main training script for IMPALA algorithm on CartPole-v1 environment."""

from dotenv import load_dotenv

from configs.paths import Paths
from training import CheckpointManager, Trainer, create_algorithm_config
from training.config import ENVIRONMENT


# Training hyperparameters
NUM_ITERATIONS = 100
CHECKPOINT_INTERVAL = 10


def main() -> None:
    """Main training function."""
    # Load environment variables
    load_dotenv(Paths.ENV_FILE)
    
    # Setup checkpoint management
    checkpoint_manager = CheckpointManager.from_env()
    
    # Create and build algorithm
    print(f"Training IMPALA on {ENVIRONMENT}...")
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


if __name__ == "__main__":
    main()