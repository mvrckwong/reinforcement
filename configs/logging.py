"""
Logging configuration settings.

Usage:
    from configs.logging import LoggingConfig, get_logging_config
    
    config = get_logging_config()
    print(config.level)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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


@lru_cache(maxsize=1)
def get_logging_config() -> LoggingConfig:
    """Get the logging configuration."""
    return LoggingConfig()


if __name__ == "__main__":
    pass
