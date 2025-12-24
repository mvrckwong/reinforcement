"""Evaluation script for trained IMPALA algorithm on CartPole-v1 environment."""

import argparse

from dotenv import load_dotenv
from loguru import logger

from configs.paths import get_paths
from configs.run_context import RunContext
from services.logs import get_logging_manager
from training import (
    CheckpointLoader,
    Evaluator,
    create_algorithm_config,
    get_algorithm_settings,
    get_evaluation_settings,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    eval_settings = get_evaluation_settings()
    
    parser = argparse.ArgumentParser(
        description="Evaluate trained IMPALA model on CartPole-v1"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to specific checkpoint to evaluate (optional, uses latest if not provided)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=eval_settings.num_episodes,
        help=f"Number of episodes to evaluate (default: {eval_settings.num_episodes})",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=eval_settings.render,
        help="Render the environment during evaluation",
    )
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List available local checkpoints and exit",
    )
    
    return parser.parse_args()


def list_available_checkpoints(checkpoint_loader: CheckpointLoader) -> None:
    """List all available local checkpoints.
    
    Args:
        checkpoint_loader: CheckpointLoader instance
    """
    checkpoints = checkpoint_loader.find_local_checkpoints()
    
    if not checkpoints:
        logger.warning("No checkpoints found.")
        logger.info("Please train a model first using train.py")
        return
    
    logger.info(f"\nFound {len(checkpoints)} checkpoint(s):")
    logger.info("=" * 80)
    
    for i, checkpoint_path in enumerate(checkpoints, 1):
        timestamp = checkpoint_path.name
        logger.info(f"{i}. {checkpoint_path}")
        logger.info(f"   Timestamp: {timestamp}")


def main() -> None:
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Load environment variables
    load_dotenv(get_paths().env_file)
    
    # Load settings
    algo_settings = get_algorithm_settings()
    
    # Setup logging
    context = RunContext(model_name="impala_cartpole_eval")
    get_logging_manager().setup(context)
    
    # Setup checkpoint loader
    checkpoint_loader = CheckpointLoader.from_env()
    
    # List checkpoints if requested
    if args.list_checkpoints:
        list_available_checkpoints(checkpoint_loader)
        return
    
    # Create algorithm config (needed for restoration)
    config = create_algorithm_config(algo_settings)
    
    # Load checkpoint
    try:
        logger.info(f"Evaluating IMPALA on {algo_settings.environment}...")
        
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            logger.info("Loading latest checkpoint...")
            checkpoint_path = None
        
        algo = checkpoint_loader.load_latest_checkpoint(
            config=config,
            checkpoint_path=checkpoint_path,
        )
        
        logger.success("Checkpoint loaded successfully")
        
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.info("\nAvailable local checkpoints:")
        list_available_checkpoints(checkpoint_loader)
        logger.info("\nTo create checkpoints:")
        logger.info("   1. Run training: python src/train.py")
        logger.info("   2. Or specify a checkpoint path: python src/evaluate.py --checkpoint <path>")
        return
    except NotImplementedError as e:
        logger.error(f"Error: {e}")
        logger.info("\nNote: S3 checkpoint loading is not yet implemented.")
        logger.info("   Checkpoints are uploaded to S3 during training but evaluation")
        logger.info("   requires local checkpoints.")
        logger.info("\n   Options:")
        logger.info("   1. Disable S3 in training (remove S3_ENDPOINT_URL from .env)")
        logger.info("   2. Run training again to create local checkpoints")
        logger.info("   3. Download checkpoint from S3 manually and use --checkpoint")
        return
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        logger.error(f"\n   Error type: {type(e).__name__}")
        return
    
    try:
        # Create evaluator and run evaluation
        evaluator = Evaluator(
            algo=algo,
            num_episodes=args.num_episodes,
            render=args.render,
        )
        
        stats = evaluator.evaluate()
        evaluator.print_statistics(stats)
        
    finally:
        algo.stop()


if __name__ == "__main__":
    main()
