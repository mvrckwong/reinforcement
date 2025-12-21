"""
S3 client creation and validation.

Usage:
    from services.s3 import get_s3_client, validate_bucket
    
    client = get_s3_client()
    if validate_bucket(client, "my-bucket"):
        print("Bucket is accessible")
"""

from functools import lru_cache
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


def validate_bucket(client: S3ClientProtocol, bucket_name: str) -> bool:
    """Check if bucket exists and is accessible.
    
    Args:
        client: S3 client instance.
        bucket_name: Name of the bucket to validate.
        
    Returns:
        True if bucket is accessible, False otherwise.
    """
    try:
        client.head_bucket(Bucket=bucket_name)
        return True
    except ClientError:
        return False


@lru_cache(maxsize=1)
def get_s3_client() -> BaseClient:
    """Get singleton S3 client instance.
    
    Returns:
        Cached S3 client instance.
    """
    return create_s3_client()


if __name__ == "__main__":
    pass

