"""
Logging manager for configuring loguru.

Usage:
    from services.logs import get_logging_manager
    from configs.run_context import RunContext
    
    context = RunContext(model_name="impala_cartpole")
    manager = get_logging_manager()
    manager.setup(context)
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from loguru import logger

from configs.logging import LoggingConfig, get_logging_config
from configs.paths import get_paths
from configs.run_context import RunContext


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


@lru_cache(maxsize=1)
def get_logging_manager() -> LoggingManager:
    """Get the logging manager."""
    return LoggingManager()


if __name__ == "__main__":
    pass