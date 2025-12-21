"""
Run context for correlating training artifacts.

Usage:
    from configs.run_context import RunContext
    
    context = RunContext(model_name="impala_cartpole")
    # context.run_id is auto-generated timestamp
    # context.run_prefix -> "impala_cartpole/20251220_193700"
    
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

import pendulum
from pydantic import BaseModel, ConfigDict, Field, computed_field


class RunContext(BaseModel):
    """Identifies a training run. Shared across logging and checkpoints."""
    
    model_name: str = Field(
        description="Name of the model being trained."
    )
    run_id: str = Field(
        default_factory=lambda: RunContext._generate_run_id(),
        description="Unique identifier for this training run."
    )
    
    # Pydantic configuration
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )
    
    @staticmethod
    def _generate_run_id() -> str:
        """Generate a timestamp-based run ID."""
        return pendulum.now().format("YYYYMMDD_HHmmss")
    
    @computed_field
    @property
    def run_prefix(self) -> str:
        """S3 prefix for all run artifacts: model_name/run_id"""
        return f"{self.model_name}/{self.run_id}"
    
    # --- Checkpoint paths ---
    
    @computed_field
    @property
    def checkpoints_prefix(self) -> str:
        """S3 prefix for checkpoints: model_name/run_id/checkpoints"""
        return f"{self.run_prefix}/checkpoints"
    
    @computed_field
    @property
    def checkpoint_best_prefix(self) -> str:
        """S3 prefix for best checkpoint: model_name/run_id/checkpoints/best"""
        return f"{self.checkpoints_prefix}/best"
    
    @computed_field
    @property
    def checkpoint_latest_prefix(self) -> str:
        """S3 prefix for latest checkpoint: model_name/run_id/checkpoints/latest"""
        return f"{self.checkpoints_prefix}/latest"
    
    @computed_field
    @property
    def checkpoint_final_prefix(self) -> str:
        """S3 prefix for final checkpoint: model_name/run_id/checkpoints/final"""
        return f"{self.checkpoints_prefix}/final"
    
    # --- Log paths ---
    
    @computed_field
    @property
    def logs_prefix(self) -> str:
        """S3 prefix for logs: model_name/run_id/logs"""
        return f"{self.run_prefix}/logs"
    
    @computed_field
    @property
    def log_file_key(self) -> str:
        """S3 key for the training log file."""
        return f"{self.logs_prefix}/training.log"
    
    # --- Metadata paths ---
    
    @computed_field
    @property
    def metadata_key(self) -> str:
        """S3 key for run metadata JSON."""
        return f"{self.run_prefix}/metadata.json"
    
    # --- Local paths (for fallback) ---
    
    @computed_field
    @property
    def local_log_filename(self) -> str:
        """Local log filename: model_name/run_id.log"""
        return f"{self.run_prefix}.log"
    
    def __str__(self) -> str:
        return self.run_prefix


if __name__ == "__main__":
    pass