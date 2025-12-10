"""Checkpoint management for training algorithms."""

import shutil
import tempfile
from pathlib import Path
from os import getenv

import pendulum
from ray.rllib.algorithms.algorithm import Algorithm
from tqdm import tqdm

from configs.paths import Paths
from utils.s3_upload import S3Uploader


CHECKPOINT_PREFIX = "impala_cartpole"
DEFAULT_BUCKET_NAME = "model"


class CheckpointManager:
    """Manages checkpoint saving for training algorithms."""
    
    def __init__(self, use_s3: bool = False, checkpoint_prefix: str = CHECKPOINT_PREFIX):
        """Initialize checkpoint manager.
        
        Args:
            use_s3: Whether to use S3 for checkpoints
            checkpoint_prefix: Prefix for checkpoint directories
        """
        self.use_s3 = use_s3
        self.checkpoint_prefix = checkpoint_prefix
        self.s3_uploader: S3Uploader | None = None
        self.checkpoint_dir: Path | None = None
        
        self._setup_storage()
    
    def _setup_storage(self) -> None:
        """Setup checkpoint storage (S3 or local)."""
        if self.use_s3:
            self.s3_uploader = S3Uploader()
            print("S3 upload configured - checkpoints will be uploaded to S3")
        else:
            self.checkpoint_dir = (
                Paths.CHECKPOINTS_DIR 
                / self.checkpoint_prefix 
                / pendulum.now().format("YYYYMMDD_HHmmss")
            )
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            print(f"Checkpoints will be saved to: {self.checkpoint_dir}")
    
    def save_checkpoint(
        self, 
        algo: Algorithm, 
        is_final: bool = False,
        verbose: bool = True,
    ) -> bool:
        """Save algorithm checkpoint.
        
        Args:
            algo: The algorithm instance to save
            is_final: Whether this is the final checkpoint
            verbose: Whether to print status messages
            
        Returns:
            True if save successful, False otherwise
        """
        if self.use_s3:
            return self._save_to_s3(algo, is_final, verbose)
        else:
            return self._save_locally(algo, verbose)
    
    def _save_to_s3(
        self, 
        algo: Algorithm, 
        is_final: bool = False,
        verbose: bool = True,
    ) -> bool:
        """Save algorithm checkpoint to S3.
        
        Args:
            algo: The algorithm instance to save
            is_final: Whether this is the final checkpoint
            verbose: Whether to print status messages
            
        Returns:
            True if upload successful, False otherwise
        """
        if not self.s3_uploader:
            return False
            
        prefix = f"{self.checkpoint_prefix}_{'final_' if is_final else ''}"
        temp_checkpoint = tempfile.mkdtemp(prefix=prefix)
        
        try:
            algo.save(temp_checkpoint)
            
            bucket_name = getenv('S3_BUCKET_NAME', DEFAULT_BUCKET_NAME)
            timestamp = pendulum.now().format("YYYYMMDD_HHmmss")
            s3_prefix = f"{self.checkpoint_prefix}/{timestamp}{'_final' if is_final else ''}"
            
            success = self.s3_uploader.upload_directory(
                temp_checkpoint, 
                bucket_name, 
                s3_prefix
            )
            
            if verbose:
                if success:
                    message = "✓ Final checkpoint uploaded to S3" if is_final else "✓ Checkpoint uploaded to S3"
                    if is_final:
                        print(message)
                    else:
                        tqdm.write(message)
                else:
                    message = "✗ Final checkpoint S3 upload failed" if is_final else "✗ S3 upload failed"
                    if is_final:
                        print(message)
                    else:
                        tqdm.write(message)
            
            return success
        finally:
            shutil.rmtree(temp_checkpoint, ignore_errors=True)
    
    def _save_locally(self, algo: Algorithm, verbose: bool = True) -> bool:
        """Save algorithm checkpoint locally.
        
        Args:
            algo: The algorithm instance to save
            verbose: Whether to print status messages
            
        Returns:
            True if save successful, False otherwise
        """
        if not self.checkpoint_dir:
            return False
            
        try:
            checkpoint_path = algo.save(str(self.checkpoint_dir))
            if verbose:
                tqdm.write(f"Checkpoint saved at: {checkpoint_path}")
            return True
        except Exception as e:
            if verbose:
                tqdm.write(f"✗ Failed to save checkpoint: {e}")
            return False
    
    @classmethod
    def from_env(cls, checkpoint_prefix: str = CHECKPOINT_PREFIX) -> "CheckpointManager":
        """Create CheckpointManager from environment configuration.
        
        Args:
            checkpoint_prefix: Prefix for checkpoint directories
            
        Returns:
            Configured CheckpointManager instance
        """
        use_s3 = bool(getenv('S3_ENDPOINT_URL'))
        return cls(use_s3=use_s3, checkpoint_prefix=checkpoint_prefix)

