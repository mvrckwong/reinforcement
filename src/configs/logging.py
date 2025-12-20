"""
Loguru logging configuration. Singleton pattern.

Usage:
    from configs.logging import setup_logging, upload_logs, logger
    from configs.run_context import RunContext
    
    context = RunContext(model_name="impala_cartpole")
    setup_logging(context=context)
    logger.info("Application started")
    
    # Upload logs to S3 when done
    upload_logs()
"""

from __future__ import annotations

import sys
from functools import lru_cache
from typing import Literal, TYPE_CHECKING

from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from configs.paths import get_paths, get_s3_paths

if TYPE_CHECKING:
    from configs.run_context import RunContext

# Module-level storage for current run context
_current_context: RunContext | None = None


class LoggingConfig(BaseSettings):
    """Logging configuration. All fields can be set via environment variables."""
    
    log_level: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    log_retention: str = Field(default="7 days")
    log_serialize: bool = Field(default=False)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def _get_config() -> LoggingConfig:
    return LoggingConfig()


def setup_logging(context: RunContext, config: LoggingConfig | None = None) -> None:
    """Configure loguru logger. Call once at application startup.
    
    Args:
        context: Run context identifying the training run
        config: Logging configuration (uses defaults if not provided)
    """
    global _current_context
    _current_context = context
    
    if config is None:
        config = _get_config()
    
    paths = get_paths()
    logger.remove()
    
    # Create model-specific log directory
    log_path = paths.logs_dir / context.log_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        sys.stderr,
        level=config.log_level,
        colorize=True,
    )
    
    logger.add(
        log_path,
        level=config.log_level,
        retention=config.log_retention,
        serialize=config.log_serialize,
    )


def upload_logs(verbose: bool = True) -> bool:
    """
    Upload current run's log file to S3.
    
    Args:
        verbose: Print status messages.
        
    Returns:
        True if upload succeeded, False otherwise.
    """
    from utils.s3_upload import S3Uploader
    
    if _current_context is None:
        if verbose:
            logger.warning("No logging context set. Call setup_logging() first.")
        return False
    
    paths = get_paths()
    s3_paths = get_s3_paths()
    
    log_file = paths.logs_dir / _current_context.log_filename
    if not log_file.exists():
        if verbose:
            logger.warning(f"Log file not found: {log_file}")
        return False
    
    uploader = S3Uploader()
    s3_key = _current_context.log_filename
    success = uploader.upload_file(
        local_path=log_file,
        bucket_name=s3_paths.logs_bucket_name,
        s3_key=s3_key,
        verbose=verbose,
    )
    
    if success and verbose:
        logger.success(f"Log uploaded to s3://{s3_paths.logs_bucket_name}/{s3_key}")
    
    return success


if __name__ == "__main__":
    pass