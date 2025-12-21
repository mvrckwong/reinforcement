"""
S3 upload service.

Usage:
    from services.s3 import get_s3_uploader
    
    uploader = get_s3_uploader()
    uploader.upload_file(local_path, bucket, s3_key)
    uploader.upload_directory(local_dir, bucket, s3_prefix)
"""

from services.s3.client import (
    S3ClientProtocol,
    create_s3_client,
    get_s3_client,
    validate_bucket,
)
from services.s3.operations import (
    validate_local_file,
    validate_local_directory,
    collect_upload_tasks,
    upload_file,
    upload_files_concurrently,
)
from services.s3.service import (
    S3Uploader,
    get_s3_uploader,
)

__all__ = [
    # Client
    "S3ClientProtocol",
    "create_s3_client",
    "get_s3_client",
    "validate_bucket",
    # Operations
    "validate_local_file",
    "validate_local_directory",
    "collect_upload_tasks",
    "upload_file",
    "upload_files_concurrently",
    # Service
    "S3Uploader",
    "get_s3_uploader",
]
