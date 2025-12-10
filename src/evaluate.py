"""Evaluation script for trained IMPALA algorithm on CartPole-v1 environment."""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from configs.paths import Paths
from training import CheckpointLoader, Evaluator, create_algorithm_config
from training.config import ENVIRONMENT


# Evaluation hyperparameters
NUM_EVAL_EPISODES = 100
RENDER = False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
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
        default=NUM_EVAL_EPISODES,
        help=f"Number of episodes to evaluate (default: {NUM_EVAL_EPISODES})",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=RENDER,
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
        print("No checkpoints found.")
        print(f"Please train a model first using train.py")
        return
    
    print(f"\nFound {len(checkpoints)} checkpoint(s):")
    print("=" * 80)
    
    for i, checkpoint_path in enumerate(checkpoints, 1):
        timestamp = checkpoint_path.name
        print(f"{i}. {checkpoint_path}")
        print(f"   Timestamp: {timestamp}")
        print()


def main() -> None:
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Load environment variables
    load_dotenv(Paths.ENV_FILE)
    
    # Setup checkpoint loader
    checkpoint_loader = CheckpointLoader.from_env()
    
    # List checkpoints if requested
    if args.list_checkpoints:
        list_available_checkpoints(checkpoint_loader)
        return
    
    # Create algorithm config (needed for restoration)
    config = create_algorithm_config()
    
    # Load checkpoint
    try:
        print(f"Evaluating IMPALA on {ENVIRONMENT}...")
        
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            print("Loading latest checkpoint...")
            checkpoint_path = None
        
        algo = checkpoint_loader.load_latest_checkpoint(
            config=config,
            checkpoint_path=checkpoint_path,
        )
        
        print("✓ Checkpoint loaded successfully")
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nAvailable local checkpoints:")
        list_available_checkpoints(checkpoint_loader)
        print("\nℹ️  To create checkpoints:")
        print("   1. Run training: python src/train.py")
        print("   2. Or specify a checkpoint path: python src/evaluate.py --checkpoint <path>")
        return
    except NotImplementedError as e:
        print(f"✗ Error: {e}")
        print("\nℹ️  Note: S3 checkpoint loading is not yet implemented.")
        print("   Checkpoints are uploaded to S3 during training but evaluation")
        print("   requires local checkpoints.")
        print("\n   Options:")
        print("   1. Disable S3 in training (remove S3_ENDPOINT_URL from .env)")
        print("   2. Run training again to create local checkpoints")
        print("   3. Download checkpoint from S3 manually and use --checkpoint")
        return
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        print(f"\n   Error type: {type(e).__name__}")
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

