"""
Logging service with S3 upload support.

Usage:
    from services.logs import get_logging_manager, upload_run_logs
    from configs.run_context import RunContext
    
    context = RunContext(model_name="impala_cartpole")
    manager = get_logging_manager()
    manager.setup(context)
    
    # ... do work ...
    
    # Upload logs to S3 when done
    upload_run_logs(context)
"""

from services.logs.manager import get_logging_manager
from services.logs.upload import upload_run_logs

__all__ = [
    "get_logging_manager",
    "upload_run_logs",
]


