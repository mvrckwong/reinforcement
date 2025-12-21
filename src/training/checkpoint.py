"""Checkpoint management for training algorithms.

Supports three checkpoint types:
    - best: Saved when reward improves (overwrites previous best)
    - latest: Saved at regular intervals (overwrites previous latest)
    - final: Saved once when training completes

S3 Structure:
    artifacts/
    └── {model_name}/
        └── {run_id}/
            ├── checkpoints/
            │   ├── best/
            │   ├── latest/
            │   └── final/
            ├── logs/
            │   └── training.log
            └── metadata.json
"""

from __future__ import annotations

import json
import shutil
import tempfile
from enum import Enum
from pathlib import Path
from os import getenv
from typing import TYPE_CHECKING, Any

from ray.rllib.algorithms.algorithm import Algorithm
from tqdm import tqdm

from configs.paths import get_paths, get_s3_paths
from services.s3 import get_s3_uploader, validate_bucket

if TYPE_CHECKING:
    from configs.run_context import RunContext


class CheckpointType(str, Enum):
    """Types of checkpoints supported."""
    BEST = "best"
    LATEST = "latest"
    FINAL = "final"


class CheckpointManager:
    """Manages checkpoint saving with best/latest/final strategy.
    
    Tries S3 first, falls back to local storage.
    """
    
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
        
        # Track best reward for smart checkpoint saving
        self.best_reward: float = float('-inf')
        
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
            return validate_bucket(uploader.client, self.s3_paths.artifacts_bucket)
        except Exception:
            return False
    
    def _log_storage_mode(self) -> None:
        """Log which storage mode is being used."""
        if self._s3_available:
            print(f"✓ S3 available - artifacts will be saved to s3://{self.s3_paths.artifacts_bucket}/{self.context.run_prefix}")
        else:
            print("✗ S3 not available - using local storage")
            self._setup_local_dir()
    
    def _setup_local_dir(self) -> Path:
        """Setup local checkpoint directory."""
        if self._local_dir is None:
            self._local_dir = get_paths().checkpoints_dir / self.context.run_prefix
            self._local_dir.mkdir(parents=True, exist_ok=True)
            print(f"Checkpoints will be saved to: {self._local_dir}")
        return self._local_dir
    
    def _get_s3_prefix(self, checkpoint_type: CheckpointType) -> str:
        """Get S3 prefix for a checkpoint type."""
        if checkpoint_type == CheckpointType.BEST:
            return self.context.checkpoint_best_prefix
        elif checkpoint_type == CheckpointType.LATEST:
            return self.context.checkpoint_latest_prefix
        else:
            return self.context.checkpoint_final_prefix
    
    def save_checkpoint(
        self, 
        algo: Algorithm, 
        checkpoint_type: CheckpointType = CheckpointType.LATEST,
        is_verbose: bool = True,
    ) -> bool:
        """Save algorithm checkpoint.
        
        Args:
            algo: The algorithm instance to save
            checkpoint_type: Type of checkpoint (best, latest, final)
            is_verbose: Whether to print status messages
            
        Returns:
            True if save successful, False otherwise
        """
        if self._s3_available:
            success = self._save_to_s3(algo, checkpoint_type, is_verbose)
            if success:
                return True
            # S3 failed, fall back to local
            if is_verbose:
                tqdm.write("S3 upload failed, falling back to local storage")
            self._s3_available = False
            self._setup_local_dir()
        
        return self._save_locally(algo, checkpoint_type, is_verbose)
    
    def save_if_best(
        self, 
        algo: Algorithm, 
        current_reward: float,
        is_verbose: bool = True,
    ) -> bool:
        """Save checkpoint only if current reward is better than previous best.
        
        Args:
            algo: The algorithm instance to save
            current_reward: Current episode reward mean
            is_verbose: Whether to print status messages
            
        Returns:
            True if checkpoint was saved (new best), False otherwise
        """
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            if is_verbose:
                tqdm.write(f"★ New best reward: {current_reward:.2f}")
            return self.save_checkpoint(algo, CheckpointType.BEST, is_verbose)
        return False
    
    def save_latest(
        self, 
        algo: Algorithm,
        is_verbose: bool = True,
    ) -> bool:
        """Save as latest checkpoint (for regular intervals).
        
        Args:
            algo: The algorithm instance to save
            is_verbose: Whether to print status messages
            
        Returns:
            True if save successful, False otherwise
        """
        return self.save_checkpoint(algo, CheckpointType.LATEST, is_verbose)
    
    def save_final(
        self, 
        algo: Algorithm,
        is_verbose: bool = True,
    ) -> bool:
        """Save final checkpoint when training completes.
        
        Args:
            algo: The algorithm instance to save
            is_verbose: Whether to print status messages
            
        Returns:
            True if save successful, False otherwise
        """
        return self.save_checkpoint(algo, CheckpointType.FINAL, is_verbose)
    
    def save_metadata(
        self, 
        metadata: dict[str, Any],
        is_verbose: bool = True,
    ) -> bool:
        """Save run metadata as JSON.
        
        Args:
            metadata: Dictionary of metadata to save
            is_verbose: Whether to print status messages
            
        Returns:
            True if save successful, False otherwise
        """
        if self._s3_available:
            return self._save_metadata_to_s3(metadata, is_verbose)
        return self._save_metadata_locally(metadata, is_verbose)
    
    def _save_to_s3(
        self, 
        algo: Algorithm, 
        checkpoint_type: CheckpointType,
        is_verbose: bool = True,
    ) -> bool:
        """Save checkpoint directly to S3 using temp directory."""
        s3_prefix = self._get_s3_prefix(checkpoint_type)
        temp_dir = tempfile.mkdtemp(prefix=f"{self.context.model_name}_{checkpoint_type.value}_")
        
        # Clean before overwrite for best/latest (they get replaced)
        clean_first = checkpoint_type in (CheckpointType.BEST, CheckpointType.LATEST)
        
        try:
            # Save to temp directory
            algo.save(temp_dir)
            
            # Upload to S3 (clean existing objects first for overwrites)
            uploader = get_s3_uploader()
            success = uploader.upload_directory(
                temp_dir,
                self.s3_paths.artifacts_bucket,
                s3_prefix,
                clean_first=clean_first,
            )
            
            if is_verbose and success:
                msg = f"✓ Checkpoint [{checkpoint_type.value}] → s3://{self.s3_paths.artifacts_bucket}/{s3_prefix}"
                tqdm.write(msg) if checkpoint_type != CheckpointType.FINAL else print(msg)
            
            return success
        except Exception as e:
            if is_verbose:
                tqdm.write(f"✗ S3 save failed: {e}")
            return False
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _save_locally(
        self, 
        algo: Algorithm, 
        checkpoint_type: CheckpointType,
        is_verbose: bool = True,
    ) -> bool:
        """Save algorithm checkpoint locally."""
        local_dir = self._setup_local_dir()
        checkpoint_dir = local_dir / checkpoint_type.value
        
        # Remove existing checkpoint of this type (overwrite)
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            checkpoint_path = algo.save(str(checkpoint_dir))
            if is_verbose:
                tqdm.write(f"✓ Checkpoint [{checkpoint_type.value}] → {checkpoint_path}")
            return True
        except Exception as e:
            if is_verbose:
                tqdm.write(f"✗ Failed to save checkpoint: {e}")
            return False
    
    def _save_metadata_to_s3(
        self, 
        metadata: dict[str, Any],
        is_verbose: bool = True,
    ) -> bool:
        """Save metadata JSON to S3."""
        temp_file = Path(tempfile.mktemp(suffix='.json'))
        
        try:
            # Write JSON to temp file
            temp_file.write_text(json.dumps(metadata, indent=2, default=str))
            
            # Upload to S3
            uploader = get_s3_uploader()
            success = uploader.upload_file(
                temp_file,
                self.s3_paths.artifacts_bucket,
                self.context.metadata_key,
            )
            
            if is_verbose and success:
                print(f"✓ Metadata → s3://{self.s3_paths.artifacts_bucket}/{self.context.metadata_key}")
            
            return success
        except Exception as e:
            if is_verbose:
                print(f"✗ Failed to save metadata: {e}")
            return False
        finally:
            temp_file.unlink(missing_ok=True)
    
    def _save_metadata_locally(
        self, 
        metadata: dict[str, Any],
        is_verbose: bool = True,
    ) -> bool:
        """Save metadata JSON locally."""
        local_dir = self._setup_local_dir()
        metadata_path = local_dir / "metadata.json"
        
        try:
            metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
            if is_verbose:
                print(f"✓ Metadata → {metadata_path}")
            return True
        except Exception as e:
            if is_verbose:
                print(f"✗ Failed to save metadata: {e}")
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
