"""
Run context for correlating training artifacts.

Usage:
    from configs.run_context import RunContext
    
    context = RunContext(model_name="impala_cartpole")
    # context.run_id is auto-generated timestamp
    # context.subpath -> "impala_cartpole/20251220_193700"
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
    def subpath(self) -> str:
        """Subpath for artifacts: model_name/run_id"""
        return f"{self.model_name}/{self.run_id}"
    
    @computed_field
    @property
    def log_filename(self) -> str:
        """Log filename: model_name/run_id.log"""
        return f"{self.subpath}.log"
    
    def __str__(self) -> str:
        return self.subpath


if __name__ == "__main__":
    pass