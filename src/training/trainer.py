"""Training loop implementation for reinforcement learning algorithms."""

from ray.rllib.algorithms.algorithm import Algorithm
from tqdm import tqdm

from training.checkpoint import CheckpointManager
from training.metrics import print_training_metrics


class Trainer:
    """Handles the training loop for RL algorithms."""
    
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
            checkpoint_interval: Save checkpoint every N iterations
        """
        self.algo = algo
        self.checkpoint_manager = checkpoint_manager
        self.num_iterations = num_iterations
        self.checkpoint_interval = checkpoint_interval
    
    def train(self) -> None:
        """Run the training loop."""
        try:
            self._training_loop()
            self._save_final_checkpoint()
        finally:
            self.algo.stop()
    
    def _training_loop(self) -> None:
        """Execute the main training loop."""
        for i in tqdm(range(self.num_iterations), desc="Training", unit="iter"):
            result = self.algo.train()
            print_training_metrics(i + 1, result)
            
            # Save periodic checkpoints
            if (i + 1) % self.checkpoint_interval == 0:
                self.checkpoint_manager.save_checkpoint(self.algo, is_final=False)
    
    def _save_final_checkpoint(self) -> None:
        """Save the final checkpoint after training completes."""
        self.checkpoint_manager.save_checkpoint(self.algo, is_final=True, is_verbose=True)


if __name__ == "__main__":
    pass