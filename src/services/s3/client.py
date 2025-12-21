"""
S3 client creation and core operations.

Usage:
    from services.s3 import get_s3_client
    
    client = get_s3_client()
"""

from functools import lru_cache
from pathlib import Path
from typing import Protocol

from boto3 import client as boto3_client
from botocore.client import Config, BaseClient
from botocore.exceptions import ClientError

from configs.storage import S3Settings


class S3ClientProtocol(Protocol):
    """Protocol for S3 client operations (DIP - depend on abstraction)."""
    
    def head_bucket(self, Bucket: str) -> dict: ...
    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None: ...


def create_s3_client(settings: S3Settings | None = None) -> BaseClient:
    """Create configured S3 client.
    
    Args:
        settings: S3 settings. If None, loads from environment.
        
    Returns:
        Configured boto3 S3 client.
    """
    settings = settings or S3Settings()
    return boto3_client(
        's3',
        endpoint_url=settings.endpoint_url,
        aws_access_key_id=settings.access_key_id,
        aws_secret_access_key=settings.secret_access_key,
        region_name=settings.region,
        config=Config(
            signature_version='s3v4',
            retries={'max_attempts': 3, 'mode': 'adaptive'},
            connect_timeout=5,
            read_timeout=60
        ),
        use_ssl=settings.use_ssl
    )


@lru_cache(maxsize=1)
def get_s3_client() -> BaseClient:
    """Get singleton S3 client instance."""
    return create_s3_client()


def validate_bucket(client: S3ClientProtocol, bucket_name: str) -> bool:
    """Check if bucket exists and is accessible."""
    try:
        client.head_bucket(Bucket=bucket_name)
        return True
    except ClientError:
        return False


def upload_file(
    client: S3ClientProtocol,
    local_path: Path,
    bucket_name: str,
    s3_key: str,
) -> bool:
    """Upload single file. Returns True on success."""
    try:
        client.upload_file(str(local_path), bucket_name, s3_key)
        return True
    except Exception:
        return False


def delete_prefix(
    client: BaseClient,
    bucket_name: str,
    prefix: str,
) -> tuple[int, str]:
    """Delete all objects under an S3 prefix.
    
    Returns:
        Tuple of (deleted_count, error_message).
    """
    try:
        paginator = client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        
        deleted_count = 0
        for page in pages:
            objects = page.get('Contents', [])
            if not objects:
                continue
            
            delete_keys = [{'Key': obj['Key']} for obj in objects]
            client.delete_objects(
                Bucket=bucket_name,
                Delete={'Objects': delete_keys}
            )
            deleted_count += len(delete_keys)
        
        return (deleted_count, "")
    except Exception as e:
        return (0, str(e))


if __name__ == "__main__":
    pass