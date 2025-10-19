"""
IMPALA Training Script for CartPole-v1

This script trains an IMPALA (Importance Weighted Actor-Learner Architecture) agent
on the CartPole-v1 environment using Ray RLlib.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from ray.rllib.algorithms.impala import IMPALAConfig
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Train IMPALA agent on CartPole-v1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Training parameters
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        help="Gymnasium environment name",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--num-env-runners",
        type=int,
        default=4,
        help="Number of parallel environment runners",
    )
    
    # Hyperparameters
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005,
        help="Learning rate",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=512,
        help="Training batch size",
    )
    
    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=5,
        help="Save checkpoint every N iterations (0 to disable)",
    )
    parser.add_argument(
        "--restore-from",
        type=str,
        default=None,
        help="Path to checkpoint to restore from",
    )
    
    return parser.parse_args()


def create_config(args: argparse.Namespace) -> IMPALAConfig:
    """Create IMPALA configuration.
    
    Args:
        args: Command-line arguments
        
    Returns:
        IMPALAConfig: Configured IMPALA algorithm
    """
    config = (
        IMPALAConfig()
        .environment(args.env)
        .learners(num_learners=0)  # Local mode for Windows compatibility
        .env_runners(num_env_runners=args.num_env_runners)
        .training(
            gamma=args.gamma,
            lr=args.lr,
            train_batch_size=args.train_batch_size,
        )
    )
    
    return config


def train_agent(args: argparse.Namespace) -> None:
    """Train the IMPALA agent.
    
    Args:
        args: Command-line arguments
    """
    logger.info("Initializing IMPALA training...")
    logger.info(f"Environment: {args.env}")
    logger.info(f"Training iterations: {args.iterations}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Discount factor (gamma): {args.gamma}")
    logger.info(f"Train batch size: {args.train_batch_size}")
    logger.info(f"Number of environment runners: {args.num_env_runners}")
    
    # Create algorithm configuration
    config = create_config(args)
    
    # Build or restore algorithm
    if args.restore_from:
        logger.info(f"Restoring from checkpoint: {args.restore_from}")
        algo = config.build()
        algo.restore(args.restore_from)
    else:
        logger.info("Building new algorithm...")
        algo = config.build()
    
    # Setup checkpoint directory
    checkpoint_dir = None
    if args.checkpoint_freq > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = Path(args.checkpoint_dir) / f"impala_{args.env.lower()}" / timestamp
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    # Training loop
    logger.info("Starting training...")
    best_reward = float("-inf")
    
    try:
        for iteration in tqdm(range(args.iterations), desc="Training", unit="iter"):
            # Train for one iteration
            result = train_iteration(algo, iteration)
            
            # Track best performance
            current_reward = result["episode_return_mean"]
            if current_reward > best_reward:
                best_reward = current_reward
                logger.info(f"New best reward: {best_reward:.2f}")
            
            # Save checkpoint
            if checkpoint_dir and args.checkpoint_freq > 0:
                if (iteration + 1) % args.checkpoint_freq == 0:
                    checkpoint_path = checkpoint_dir / f"iter_{iteration + 1:04d}"
                    algo.save(str(checkpoint_path))
                    logger.info(f"Checkpoint saved to: {checkpoint_path}")
        
        # Save final checkpoint
        if checkpoint_dir:
            final_checkpoint = checkpoint_dir / "final"
            algo.save(str(final_checkpoint))
            logger.info(f"Final checkpoint saved to: {final_checkpoint}")
        
        logger.info("Training completed successfully!")
        logger.info(f"Best reward achieved: {best_reward:.2f}")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        if checkpoint_dir:
            interrupt_checkpoint = checkpoint_dir / "interrupted"
            algo.save(str(interrupt_checkpoint))
            logger.info(f"Checkpoint saved to: {interrupt_checkpoint}")
    
    finally:
        algo.stop()
        logger.info("Algorithm stopped and resources cleaned up")


def train_iteration(algo, iteration: int) -> Dict[str, Any]:
    """Train for one iteration and log results.
    
    Args:
        algo: RLlib algorithm instance
        iteration: Current iteration number
        
    Returns:
        Dict containing training metrics
    """
    result = algo.train()
    
    # Extract metrics
    episodes = result["env_runners"]["num_episodes"]
    reward = result["env_runners"]["episode_return_mean"]
    length = result["env_runners"]["episode_len_mean"]
    
    # Log progress
    tqdm.write(
        f"Iter {iteration + 1:2d} | "
        f"Episodes: {episodes:6.0f} | "
        f"Reward: {reward:6.2f} | "
        f"Length: {length:6.2f}"
    )
    
    return {
        "iteration": iteration + 1,
        "num_episodes": episodes,
        "episode_return_mean": reward,
        "episode_len_mean": length,
    }


def main() -> None:
    """Main entry point."""
    args = parse_arguments()
    train_agent(args)


if __name__ == "__main__":
    main()

