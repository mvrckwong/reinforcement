"""
Loguru logging configuration. Singleton pattern.

Usage:
    from configs.logging import get_logging_manager, get_logging_config, upload_run_logs, logger
    from configs.run_context import RunContext
    
    context = RunContext(model_name="impala_cartpole")
    manager = get_logging_manager()
    manager.setup(context)
    logger.info("Application started")
    
    # Upload logs to S3 when done
    upload_run_logs(context)
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from configs.paths import get_paths, get_s3_paths
from configs.run_context import RunContext


class LoggingConfig(BaseSettings):
    """Logging configuration. All fields can be set via environment variables."""
    
    level: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Minimum log level to capture."
    )
    retention: str = Field(
        default="7 days",
        alias="LOG_RETENTION",
        description="How long to retain log files."
    )
    serialize: bool = Field(
        default=False,
        alias="LOG_SERIALIZE",
        description="Whether to serialize logs as JSON."
    )
    
    # Pydantic configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
        populate_by_name=True,
    )


class LoggingManager:
    """Configures loguru logging for a training run."""
    
    def setup(
        self, 
        context: RunContext, 
        config: LoggingConfig | None = None
    ) -> Path:
        """Configure loguru logger. Call once at application startup.
        
        Args:
            context: Run context identifying the training run
            config: Logging configuration (uses defaults if not provided)
            
        Returns:
            Path to the log file.
        """
        if config is None:
            config = get_logging_config()
        
        paths = get_paths()
        logger.remove()
        
        # Create model-specific log directory (local fallback)
        log_path = paths.logs_dir / context.local_log_filename
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            sys.stderr,
            level=config.level,
            colorize=True,
        )
        
        logger.add(
            log_path,
            level=config.level,
            retention=config.retention,
            serialize=config.serialize,
        )
        
        return log_path


def upload_run_logs(
    context: RunContext, 
    is_verbose: bool = True,
) -> bool:
    """Upload a run's log file to S3.
    
    Uploads to: s3://{bucket}/{model_name}/{run_id}/logs/training.log
    
    Args:
        context: Run context identifying the training run.
        is_verbose: Print status messages.
        
    Returns:
        True if upload succeeded, False otherwise.
    """
    from services.s3 import get_s3_uploader
    
    paths = get_paths()
    s3_paths = get_s3_paths()

    # Get the log file path and check if it exists
    log_file = paths.logs_dir / context.local_log_filename
    if not log_file.exists():
        if is_verbose:
            logger.warning(f"Log file not found: {log_file}")
        return False
    
    # Upload the log file to S3 using new unified path structure
    uploader = get_s3_uploader()
    is_uploaded = uploader.upload_file(
        local_path=log_file,
        bucket_name=s3_paths.artifacts_bucket,
        s3_key=context.log_file_key,
        is_verbose=is_verbose,
    )
    
    # Log success or failure
    if is_uploaded and is_verbose:
        logger.success(
            f"Log uploaded to s3://{s3_paths.artifacts_bucket}/{context.log_file_key}"
        )
    
    return is_uploaded


@lru_cache(maxsize=1)
def get_logging_config() -> LoggingConfig:
    """Get the logging configuration."""
    return LoggingConfig()


@lru_cache(maxsize=1)
def get_logging_manager() -> LoggingManager:
    """Get the logging manager."""
    return LoggingManager()


if __name__ == "__main__":
    pass
