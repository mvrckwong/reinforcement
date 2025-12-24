"""
Log file upload operations.

Usage:
    from services.logs import upload_run_logs
    from configs.run_context import RunContext
    
    context = RunContext(model_name="impala_cartpole")
    upload_run_logs(context)
"""

from __future__ import annotations

from loguru import logger

from configs.paths import get_paths, get_s3_paths
from configs.run_context import RunContext


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

if __name__ == "__main__":
    pass