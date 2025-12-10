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
    
    def __init__(
        self, 
        use_s3: bool = False, 
        checkpoint_prefix: str = CHECKPOINT_PREFIX,
        keep_local_copy: bool = True,
    ):
        """Initialize checkpoint manager.
        
        Args:
            use_s3: Whether to upload checkpoints to S3
            checkpoint_prefix: Prefix for checkpoint directories
            keep_local_copy: Whether to keep local checkpoints (even when using S3)
        """
        self.use_s3 = use_s3
        self.checkpoint_prefix = checkpoint_prefix
        self.keep_local_copy = keep_local_copy
        self.s3_uploader: S3Uploader | None = None
        self.checkpoint_dir: Path | None = None
        
        self._setup_storage()
    
    def _setup_storage(self) -> None:
        """Setup checkpoint storage (local and optionally S3)."""
        # Always create local checkpoint directory if keeping local copy
        if self.keep_local_copy or not self.use_s3:
            self.checkpoint_dir = (
                Paths.CHECKPOINTS_DIR 
                / self.checkpoint_prefix 
                / pendulum.now().format("YYYYMMDD_HHmmss")
            )
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            print(f"Checkpoints will be saved to: {self.checkpoint_dir}")
        
        # Setup S3 if configured
        if self.use_s3:
            self.s3_uploader = S3Uploader()
            print("S3 upload configured - checkpoints will also be uploaded to S3")
    
    def save_checkpoint(
        self, 
        algo: Algorithm, 
        is_final: bool = False,
        verbose: bool = True,
    ) -> bool:
        """Save algorithm checkpoint.
        
        Saves locally (if keep_local_copy is True) and optionally uploads to S3.
        
        Args:
            algo: The algorithm instance to save
            is_final: Whether this is the final checkpoint
            verbose: Whether to print status messages
            
        Returns:
            True if save successful, False otherwise
        """
        success = True
        
        # Save locally first (if configured)
        if self.checkpoint_dir:
            success = self._save_locally(algo, verbose) and success
        
        # Upload to S3 (if configured)
        if self.use_s3:
            success = self._save_to_s3_from_local(algo, is_final, verbose) and success
        
        return success
    
    def _save_to_s3_from_local(
        self, 
        algo: Algorithm, 
        is_final: bool = False,
        verbose: bool = True,
    ) -> bool:
        """Upload checkpoint to S3.
        
        If local checkpoint directory exists, uploads it directly.
        Otherwise, creates a temp directory for the checkpoint.
        
        Args:
            algo: The algorithm instance to save
            is_final: Whether this is the final checkpoint
            verbose: Whether to print status messages
            
        Returns:
            True if upload successful, False otherwise
        """
        if not self.s3_uploader:
            return False
        
        # Use local checkpoint dir if available, otherwise create temp
        if self.checkpoint_dir:
            checkpoint_path = str(self.checkpoint_dir)
            cleanup_temp = False
        else:
            prefix = f"{self.checkpoint_prefix}_{'final_' if is_final else ''}"
            checkpoint_path = tempfile.mkdtemp(prefix=prefix)
            algo.save(checkpoint_path)
            cleanup_temp = True
        
        try:
            bucket_name = getenv('S3_BUCKET_NAME', DEFAULT_BUCKET_NAME)
            timestamp = pendulum.now().format("YYYYMMDD_HHmmss")
            s3_prefix = f"{self.checkpoint_prefix}/{timestamp}{'_final' if is_final else ''}"
            
            success = self.s3_uploader.upload_directory(
                checkpoint_path, 
                bucket_name, 
                s3_prefix
            )
            
            if verbose:
                if success:
                    message = "✓ Checkpoint uploaded to S3"
                    if is_final:
                        print(message)
                    else:
                        tqdm.write(message)
                else:
                    message = "✗ S3 upload failed"
                    if is_final:
                        print(message)
                    else:
                        tqdm.write(message)
            
            return success
        finally:
            if cleanup_temp:
                shutil.rmtree(checkpoint_path, ignore_errors=True)
    
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
    def from_env(
        cls, 
        checkpoint_prefix: str = CHECKPOINT_PREFIX,
        keep_local_copy: bool = True,
    ) -> "CheckpointManager":
        """Create CheckpointManager from environment configuration.
        
        Args:
            checkpoint_prefix: Prefix for checkpoint directories
            keep_local_copy: Whether to keep local checkpoints (even when using S3)
            
        Returns:
            Configured CheckpointManager instance
        """
        use_s3 = bool(getenv('S3_ENDPOINT_URL'))
        return cls(
            use_s3=use_s3, 
            checkpoint_prefix=checkpoint_prefix,
            keep_local_copy=keep_local_copy,
        )

