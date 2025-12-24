"""Training loop implementation for reinforcement learning algorithms."""

from typing import Any

from ray.rllib.algorithms.algorithm import Algorithm
from tqdm import tqdm

from training.checkpoint import CheckpointManager
from training.metrics import extract_metric, log_training_metrics


class Trainer:
    """Handles the training loop for RL algorithms with smart checkpointing.
    
    Checkpoint Strategy:
        - best: Saved when episode reward improves (overwrites previous)
        - latest: Saved at regular intervals (overwrites previous)
        - final: Saved once when training completes
        - metadata: Saved with final checkpoint (config, best reward, etc.)
    """
    
    def __init__(
        self,
        algo: Algorithm,
        checkpoint_manager: CheckpointManager,
        num_iterations: int = 100,
        checkpoint_interval: int = 10,
    ):
        """Initialize trainer.
        
        Args:
            algo: The algorithm instance to train
            checkpoint_manager: Manager for handling checkpoints
            num_iterations: Number of training iterations
            checkpoint_interval: Save 'latest' checkpoint every N iterations
        """
        self.algo = algo
        self.checkpoint_manager = checkpoint_manager
        self.num_iterations = num_iterations
        self.checkpoint_interval = checkpoint_interval
        
        # Track training stats for metadata
        self._total_episodes = 0
        self._total_steps = 0
        self._final_reward = 0.0
    
    def train(self) -> None:
        """Run the training loop."""
        try:
            self._training_loop()
            self._save_final_artifacts()
        finally:
            self.algo.stop()
    
    def _training_loop(self) -> None:
        """Execute the main training loop with smart checkpointing."""
        for i in tqdm(range(self.num_iterations), desc="Training", unit="iter"):
            result = self.algo.train()
            log_training_metrics(i + 1, result)
            
            # Extract current metrics
            current_reward = extract_metric(
                result, 
                "episode_return_mean", 
                default=extract_metric(result, "episode_reward_mean", 0.0)
            )
            
            # Update tracking stats
            self._total_episodes = extract_metric(
                result, 
                "num_episodes", 
                default=extract_metric(result, "episodes_total", 0)
            )
            self._total_steps = extract_metric(
                result, 
                "num_env_steps_sampled", 
                default=extract_metric(result, "env_steps_sampled", 0)
            )
            self._final_reward = current_reward
            
            # Save best checkpoint if reward improved
            self.checkpoint_manager.save_if_best(self.algo, current_reward)
            
            # Save latest checkpoint at regular intervals
            if (i + 1) % self.checkpoint_interval == 0:
                self.checkpoint_manager.save_latest(self.algo)
    
    def _save_final_artifacts(self) -> None:
        """Save final checkpoint and run metadata."""
        # Save final checkpoint
        self.checkpoint_manager.save_final(self.algo, is_verbose=True)
        
        # Save run metadata
        metadata = self._build_metadata()
        self.checkpoint_manager.save_metadata(metadata, is_verbose=True)
    
    def _build_metadata(self) -> dict[str, Any]:
        """Build metadata dictionary for the run."""
        return {
            "run_id": self.checkpoint_manager.context.run_id,
            "model_name": self.checkpoint_manager.context.model_name,
            "training": {
                "num_iterations": self.num_iterations,
                "checkpoint_interval": self.checkpoint_interval,
                "total_episodes": self._total_episodes,
                "total_steps": self._total_steps,
            },
            "results": {
                "best_reward": self.checkpoint_manager.best_reward,
                "final_reward": self._final_reward,
            },
            "paths": {
                "checkpoints_prefix": self.checkpoint_manager.context.checkpoints_prefix,
                "logs_prefix": self.checkpoint_manager.context.logs_prefix,
            },
        }


if __name__ == "__main__":
    pass
