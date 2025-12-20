"""
Loguru logging configuration. Singleton pattern.

Usage:
    from configs.logging import setup_logging, upload_logs, logger
    
    setup_logging()
    logger.info("Application started")
    
    # Upload logs to S3 when done
    upload_logs()
"""

from __future__ import annotations

import sys
from functools import lru_cache
from typing import Literal

from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from configs.paths import get_paths, get_s3_paths


class LoggingConfig(BaseSettings):
    """Logging configuration. All fields can be set via environment variables."""
    
    log_level: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    log_retention: str = Field(default="7 days")
    log_rotation: str = Field(default="1 day")
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


def setup_logging(config: LoggingConfig | None = None) -> None:
    """Configure loguru logger. Call once at application startup."""
    if config is None:
        config = _get_config()
    
    paths = get_paths()
    logger.remove()
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        sys.stderr,
        level=config.log_level,
        colorize=True,
    )
    
    logger.add(
        paths.logs_dir / "{time:YYYY-MM-DD}.log",
        level=config.log_level,
        rotation=config.log_rotation,
        retention=config.log_retention,
        compression="zip",
        serialize=config.log_serialize,
    )


def upload_logs(verbose: bool = True) -> bool:
    """
    Upload local logs to S3.
    
    Args:
        verbose: Print status messages.
        
    Returns:
        True if upload succeeded, False otherwise.
    """
    from utils.s3_upload import S3Uploader
    
    paths = get_paths()
    s3_paths = get_s3_paths()
    
    if not paths.logs_dir.exists():
        if verbose:
            logger.warning(f"Logs directory not found: {paths.logs_dir}")
        return False
    
    uploader = S3Uploader()
    success = uploader.upload_directory(
        local_dir=paths.logs_dir,
        bucket_name=s3_paths.logs_bucket_name,
        verbose=verbose,
    )
    
    if success and verbose:
        logger.success(f"Logs uploaded to s3://{s3_paths.logs_bucket_name}/")
    
    return success


if __name__ == "__main__":
    pass