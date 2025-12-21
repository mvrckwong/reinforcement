"""
S3 upload service with lifecycle management.

Usage:
    from services.s3 import get_s3_uploader, configure_lifecycle_rules
    
    uploader = get_s3_uploader()
    uploader.upload_file(local_path, bucket, s3_key)
    uploader.upload_directory(local_dir, bucket, s3_prefix, clean_first=True)
    
    # Configure lifecycle rules for automatic cleanup
    configure_lifecycle_rules(uploader.client, bucket, is_verbose=True)
"""

from services.s3.client import S3ClientProtocol, get_s3_client, validate_bucket
from services.s3.service import S3Uploader, get_s3_uploader
from services.s3.lifecycle import (
    DEFAULT_LIFECYCLE_RULES,
    configure_lifecycle_rules,
    get_lifecycle_rules,
    delete_lifecycle_rules,
)

__all__ = [
    # Client (for type hints and direct access)
    "S3ClientProtocol",
    "get_s3_client",
    "validate_bucket",
    
    # Service (main entry point)
    "S3Uploader",
    "get_s3_uploader",
    
    # Lifecycle
    "DEFAULT_LIFECYCLE_RULES",
    "configure_lifecycle_rules",
    "get_lifecycle_rules",
    "delete_lifecycle_rules",
]
