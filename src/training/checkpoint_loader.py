"""Utilities for loading checkpoints for evaluation."""

import tempfile
from pathlib import Path
from os import getenv

import pendulum
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from configs.paths import Paths
from utils.s3_upload import S3Uploader


CHECKPOINT_PREFIX = "impala_cartpole"
DEFAULT_BUCKET_NAME = "model"


class CheckpointLoader:
    """Handles loading checkpoints for evaluation."""
    
    def __init__(self, use_s3: bool = False, checkpoint_prefix: str = CHECKPOINT_PREFIX):
        """Initialize checkpoint loader.
        
        Args:
            use_s3: Whether to load from S3
            checkpoint_prefix: Prefix for checkpoint directories
        """
        self.use_s3 = use_s3
        self.checkpoint_prefix = checkpoint_prefix
        self.s3_uploader: S3Uploader | None = None
        
        if self.use_s3:
            self.s3_uploader = S3Uploader()
    
    def load_latest_checkpoint(
        self, 
        config: AlgorithmConfig,
        checkpoint_path: str | None = None,
        prefer_local: bool = True,
    ) -> Algorithm:
        """Load the latest checkpoint.
        
        Args:
            config: Algorithm configuration
            checkpoint_path: Specific checkpoint path (optional)
            prefer_local: Try loading from local first, even if S3 is configured
            
        Returns:
            Algorithm instance loaded from checkpoint
        """
        if checkpoint_path:
            print(f"Loading checkpoint from: {checkpoint_path}")
            return Algorithm.from_checkpoint(checkpoint_path)
        
        # Try local first if preferred (default for evaluation)
        if prefer_local:
            try:
                return self._load_from_local(config)
            except FileNotFoundError as e:
                # If S3 is configured, try that as fallback
                if self.use_s3:
                    print(f"No local checkpoints found. Attempting to load from S3...")
                    return self._load_from_s3(config)
                else:
                    # Re-raise if no S3 fallback available
                    raise e
        
        # Original logic: respect use_s3 setting
        if self.use_s3:
            return self._load_from_s3(config)
        else:
            return self._load_from_local(config)
    
    def _load_from_local(self, config: AlgorithmConfig) -> Algorithm:
        """Load latest checkpoint from local filesystem.
        
        Args:
            config: Algorithm configuration
            
        Returns:
            Algorithm instance loaded from checkpoint
            
        Raises:
            FileNotFoundError: If no checkpoints found
        """
        checkpoint_base = Paths.CHECKPOINTS_DIR / self.checkpoint_prefix
        
        if not checkpoint_base.exists():
            raise FileNotFoundError(
                f"No checkpoints found in {checkpoint_base}. "
                "Please train a model first."
            )
        
        # Find the latest checkpoint directory
        checkpoint_dirs = sorted(
            [d for d in checkpoint_base.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        
        if not checkpoint_dirs:
            raise FileNotFoundError(
                f"No checkpoint directories found in {checkpoint_base}. "
                "Please train a model first."
            )
        
        latest_dir = checkpoint_dirs[0]
        
        # Find checkpoint files in the directory
        checkpoint_files = list(latest_dir.glob("**/checkpoint-*"))
        
        if not checkpoint_files:
            # Try to find algorithm_state.pkl or similar
            checkpoint_files = list(latest_dir.glob("**/algorithm_state.pkl"))
        
        if not checkpoint_files:
            raise FileNotFoundError(
                f"No checkpoint files found in {latest_dir}. "
                "The checkpoint directory may be incomplete."
            )
        
        # Use the directory itself as RLlib expects
        checkpoint_path = str(latest_dir)
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        return Algorithm.from_checkpoint(checkpoint_path)
    
    def _load_from_s3(self, config: AlgorithmConfig) -> Algorithm:
        """Load latest checkpoint from S3.
        
        Args:
            config: Algorithm configuration
            
        Returns:
            Algorithm instance loaded from checkpoint
            
        Raises:
            NotImplementedError: S3 loading not yet fully implemented
        """
        # This would require:
        # 1. List all checkpoints in S3
        # 2. Find the latest one
        # 3. Download to temp directory
        # 4. Load from temp directory
        # For now, we'll raise a helpful error
        raise NotImplementedError(
            "Loading from S3 is not yet implemented. "
            "Please provide a local checkpoint path or download from S3 manually."
        )
    
    def find_local_checkpoints(self) -> list[Path]:
        """Find all local checkpoint directories.
        
        Returns:
            List of checkpoint directory paths, sorted by modification time (newest first)
        """
        checkpoint_base = Paths.CHECKPOINTS_DIR / self.checkpoint_prefix
        
        if not checkpoint_base.exists():
            return []
        
        checkpoint_dirs = sorted(
            [d for d in checkpoint_base.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        
        return checkpoint_dirs
    
    @classmethod
    def from_env(cls, checkpoint_prefix: str = CHECKPOINT_PREFIX) -> "CheckpointLoader":
        """Create CheckpointLoader from environment configuration.
        
        Note: By default, load_latest_checkpoint() will prefer local checkpoints
        even if S3 is configured. This is useful when S3 is configured for 
        training uploads but you want to evaluate local checkpoints.
        
        Args:
            checkpoint_prefix: Prefix for checkpoint directories
            
        Returns:
            Configured CheckpointLoader instance
        """
        use_s3 = bool(getenv('S3_ENDPOINT_URL'))
        return cls(use_s3=use_s3, checkpoint_prefix=checkpoint_prefix)

