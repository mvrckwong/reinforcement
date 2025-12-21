"""Checkpoint management for training algorithms."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from os import getenv
from typing import TYPE_CHECKING

import pendulum
from ray.rllib.algorithms.algorithm import Algorithm
from tqdm import tqdm

from configs.paths import get_paths, get_s3_paths
from services.s3 import get_s3_uploader

if TYPE_CHECKING:
    from configs.run_context import RunContext


class CheckpointManager:
    """Manages checkpoint saving. Tries S3 first, falls back to local."""
    
    def __init__(
        self, 
        context: RunContext,
        s3_paths=None,
    ):
        """Initialize checkpoint manager.
        
        Args:
            context: Run context identifying the training run
            s3_paths: S3 path configuration (loaded from env if not provided)
        """
        self.context = context
        self.s3_paths = s3_paths or get_s3_paths()
        
        # Try to setup S3, track if available
        self._s3_available = self._check_s3_available()
        self._local_dir: Path | None = None
        
        self._log_storage_mode()
    
    def _check_s3_available(self) -> bool:
        """Check if S3 is configured and accessible."""
        if not getenv('S3_ENDPOINT_URL'):
            return False
        
        try:
            uploader = get_s3_uploader()
            # Test bucket access
            uploader.client.head_bucket(Bucket=self.s3_paths.checkpoints_bucket_name)
            return True
        except Exception:
            return False
    
    def _log_storage_mode(self) -> None:
        """Log which storage mode is being used."""
        if self._s3_available:
            print(f"✓ S3 available - checkpoints will be saved to s3://{self.s3_paths.checkpoints_bucket_name}")
        else:
            print("✗ S3 not available - using local storage")
            self._setup_local_dir()
    
    def _setup_local_dir(self) -> Path:
        """Setup local checkpoint directory."""
        if self._local_dir is None:
            self._local_dir = get_paths().checkpoints_dir / self.context.subpath
            self._local_dir.mkdir(parents=True, exist_ok=True)
            print(f"Checkpoints will be saved to: {self._local_dir}")
        return self._local_dir
    
    def save_checkpoint(
        self, 
        algo: Algorithm, 
        is_final: bool = False,
        is_verbose: bool = True,
    ) -> bool:
        """Save algorithm checkpoint. Tries S3 first, falls back to local.
        
        Args:
            algo: The algorithm instance to save
            is_final: Whether this is the final checkpoint
            is_verbose: Whether to print status messages
            
        Returns:
            True if save successful, False otherwise
        """
        if self._s3_available:
            success = self._save_to_s3(algo, is_final, is_verbose)
            if success:
                return True
            # S3 failed, fall back to local
            if is_verbose:
                tqdm.write("S3 upload failed, falling back to local storage")
            self._s3_available = False
            self._setup_local_dir()
        
        return self._save_locally(algo, is_verbose)
    
    def _save_to_s3(
        self, 
        algo: Algorithm, 
        is_final: bool = False,
        is_verbose: bool = True,
    ) -> bool:
        """Save checkpoint directly to S3 using temp directory.
        
        Args:
            algo: The algorithm instance to save
            is_final: Whether this is the final checkpoint
            is_verbose: Whether to print status messages
            
        Returns:
            True if upload successful, False otherwise
        """
        suffix = "_final" if is_final else f"_{pendulum.now().format('HHmmss')}"
        temp_dir = tempfile.mkdtemp(prefix=f"{self.context.model_name}{suffix}_")
        
        try:
            # Save to temp directory
            algo.save(temp_dir)
            
            # Upload to S3
            s3_key = f"{self.context.subpath}{suffix}"
            
            uploader = get_s3_uploader()
            success = uploader.upload_directory(
                temp_dir,
                self.s3_paths.checkpoints_bucket_name,
                s3_key,
            )
            
            if is_verbose and success:
                msg = f"✓ Checkpoint uploaded to S3: {s3_key}"
                tqdm.write(msg) if not is_final else print(msg)
            
            return success
        except Exception as e:
            if is_verbose:
                tqdm.write(f"✗ S3 save failed: {e}")
            return False
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _save_locally(self, algo: Algorithm, is_verbose: bool = True) -> bool:
        """Save algorithm checkpoint locally.
        
        Args:
            algo: The algorithm instance to save
            is_verbose: Whether to print status messages
            
        Returns:
            True if save successful, False otherwise
        """
        local_dir = self._setup_local_dir()
        
        try:
            checkpoint_path = algo.save(str(local_dir))
            if is_verbose:
                tqdm.write(f"✓ Checkpoint saved: {checkpoint_path}")
            return True
        except Exception as e:
            if is_verbose:
                tqdm.write(f"✗ Failed to save checkpoint: {e}")
            return False
    
    @classmethod
    def from_env(cls, context: RunContext) -> "CheckpointManager":
        """Create CheckpointManager from environment configuration.
        
        Args:
            context: Run context identifying the training run
            
        Returns:
            Configured CheckpointManager instance
        """
        return cls(context=context, s3_paths=get_s3_paths())


if __name__ == "__main__":
    pass