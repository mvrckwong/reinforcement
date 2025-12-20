"""Evaluation logic for trained RL algorithms."""

from typing import Any, Mapping

from ray.rllib.algorithms.algorithm import Algorithm
from tqdm import tqdm

from training.metrics import extract_metric


class Evaluator:
    """Handles evaluation of trained RL algorithms."""
    
    def __init__(
        self,
        algo: Algorithm,
        num_episodes: int = 100,
        render: bool = False,
    ):
        """Initialize evaluator.
        
        Args:
            algo: The trained algorithm instance to evaluate
            num_episodes: Number of episodes to run for evaluation
            render: Whether to render the environment
        """
        self.algo = algo
        self.num_episodes = num_episodes
        self.render = render
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
    
    def evaluate(self) -> dict[str, float]:
        """Run evaluation and return statistics.
        
        Returns:
            Dictionary containing evaluation statistics
        """
        print(f"Evaluating for {self.num_episodes} episodes...")
        
        for episode in tqdm(range(self.num_episodes), desc="Evaluating", unit="episode"):
            episode_reward, episode_length = self._run_episode()
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            if (episode + 1) % 10 == 0:
                self._print_progress(episode + 1)
        
        return self._compute_statistics()
    
    def _run_episode(self) -> tuple[float, int]:
        """Run a single evaluation episode.
        
        Returns:
            Tuple of (total_reward, episode_length)
        """
        # Get a single worker for evaluation
        worker = self.algo.workers.local_worker()
        env = worker.env
        
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0
        
        while not (done or truncated):
            # Get action from policy
            action = self.algo.compute_single_action(obs, explore=False)
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if self.render:
                env.render()
        
        return total_reward, steps
    
    def _print_progress(self, episode: int) -> None:
        """Print progress statistics.
        
        Args:
            episode: Current episode number
        """
        recent_rewards = self.episode_rewards[-10:]
        recent_lengths = self.episode_lengths[-10:]
        
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        avg_length = sum(recent_lengths) / len(recent_lengths)
        
        tqdm.write(
            f"Episode {episode:3d} | "
            f"Avg Reward (last 10): {avg_reward:6.2f} | "
            f"Avg Length (last 10): {avg_length:6.2f}"
        )
    
    def _compute_statistics(self) -> dict[str, float]:
        """Compute final evaluation statistics.
        
        Returns:
            Dictionary with mean, std, min, max statistics
        """
        import numpy as np
        
        rewards_array = np.array(self.episode_rewards)
        lengths_array = np.array(self.episode_lengths)
        
        stats = {
            "mean_reward": float(np.mean(rewards_array)),
            "std_reward": float(np.std(rewards_array)),
            "min_reward": float(np.min(rewards_array)),
            "max_reward": float(np.max(rewards_array)),
            "mean_length": float(np.mean(lengths_array)),
            "std_length": float(np.std(lengths_array)),
            "min_length": float(np.min(lengths_array)),
            "max_length": float(np.max(lengths_array)),
        }
        
        return stats
    
    def print_statistics(self, stats: dict[str, float]) -> None:
        """Print formatted evaluation statistics.
        
        Args:
            stats: Statistics dictionary from evaluate()
        """
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Episodes: {self.num_episodes}")
        print(f"\nReward Statistics:")
        print(f"  Mean:   {stats['mean_reward']:8.2f} ± {stats['std_reward']:.2f}")
        print(f"  Min:    {stats['min_reward']:8.2f}")
        print(f"  Max:    {stats['max_reward']:8.2f}")
        print(f"\nEpisode Length Statistics:")
        print(f"  Mean:   {stats['mean_length']:8.2f} ± {stats['std_length']:.2f}")
        print(f"  Min:    {stats['min_length']:8.0f}")
        print(f"  Max:    {stats['max_length']:8.0f}")
        print("=" * 60)


if __name__ == "__main__":
    pass