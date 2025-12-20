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


class S3Paths(BaseSettings):
    """S3 path structure configuration (environment-configurable).
    
    Note: These are path structures only, not connection settings.
    For S3 credentials/endpoint, see configs.storage.S3Settings
    """
    
    bucket_name: str = Field(
        default='model', 
        alias='S3_BUCKET_NAME',
        description="S3 bucket for checkpoints"
    )
    logs_bucket_name: str = Field(
        default='logging', 
        alias='S3_LOGS_BUCKET_NAME',
        description="S3 bucket for logs"
    )
    checkpoints_prefix: str = Field(
        default='checkpoints', 
        alias='S3_CHECKPOINTS_PREFIX',
        description="S3 prefix for checkpoint storage"
    )
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore'
    )

@lru_cache(maxsize=1)
def get_paths() -> Paths:
    """Get the paths configuration."""
    return Paths()

@lru_cache(maxsize=1)
def get_s3_paths() -> S3Paths:
    """Get the S3 paths configuration."""
    return S3Paths()


if __name__ == "__main__":
    pass