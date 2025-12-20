"""
Paths configuration. Singleton pattern.

Usage:
    from configs.paths import get_paths
    
    # Get the paths
    paths = get_paths()
"""

from pathlib import Path
from functools import lru_cache

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Computed once at module load
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Paths(BaseSettings):
    """Local filesystem paths configuration."""
    
    logs_dir: Path = Field(
        default=_PROJECT_ROOT / "logs", 
        description="Logs directory."
    )
    checkpoints_dir: Path = Field(
        default=_PROJECT_ROOT / "checkpoints", 
        description="Checkpoints directory."
    )
    env_file: Path = Field(
        default=_PROJECT_ROOT / ".env", 
        description="Environment file."
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
    )
    
    @computed_field
    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT


@lru_cache(maxsize=1)
def get_paths() -> Paths:
    return Paths()


class S3Paths(BaseSettings):
    """S3 path structure configuration (environment-configurable).
    
    Note: These are path structures only, not connection settings.
    For S3 credentials/endpoint, see configs.storage.S3Settings
    """
    
    bucket_name: str = Field(
        default='model', 
        alias='S3_BUCKET_NAME',
        description="Default S3 bucket for model artifacts"
    )
    checkpoints_prefix: str = Field(
        default='checkpoints', 
        alias='S3_CHECKPOINTS_PREFIX',
        description="S3 prefix for checkpoint storage"
    )
    logs_prefix: str = Field(
        default='logging', 
        alias='S3_LOGS_PREFIX',
        description="S3 prefix for log storage"
    )
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore'
    )
    
    def checkpoint_key(self, model_name: str, timestamp: str, is_final: bool = False) -> str:
        """Build S3 key for a checkpoint.
        
        Args:
            model_name: Name/prefix of the model
            timestamp: Timestamp string (e.g., '20241216_120000')
            is_final: Whether this is the final checkpoint
            
        Returns:
            S3 key like 'checkpoints/impala_cartpole/20241216_120000'
        """
        suffix = '_final' if is_final else ''
        return f"{model_name}/{timestamp}{suffix}"
    
    def checkpoint_uri(self, model_name: str, timestamp: str, is_final: bool = False) -> str:
        """Build full S3 URI for a checkpoint.
        
        Args:
            model_name: Name/prefix of the model
            timestamp: Timestamp string
            is_final: Whether this is the final checkpoint
            
        Returns:
            S3 URI like 's3://model/checkpoints/impala_cartpole/20241216_120000'
        """
        key = self.checkpoint_key(model_name, timestamp, is_final)
        return f"s3://{self.bucket_name}/{self.checkpoints_prefix}/{key}"


if __name__ == "__main__":
    pass