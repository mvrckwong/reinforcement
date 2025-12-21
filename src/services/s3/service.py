"""
S3 upload service.

Usage:
    from services.s3 import get_s3_uploader
    
    uploader = get_s3_uploader()
    uploader.upload_file(local_path, bucket, s3_key)
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from services.s3.client import S3ClientProtocol, validate_bucket, get_s3_client
from services.s3.operations import (
    delete_prefix,
    validate_local_file,
    validate_local_directory,
    collect_upload_tasks,
    upload_file,
    upload_files_concurrently,
)


class S3Uploader:
    """Orchestrator for S3 upload operations."""
    
    def __init__(self, client: S3ClientProtocol):
        """Initialize the S3 uploader.
        
        Args:
            client: Injected S3 client (DIP - depend on abstraction).
        """
        self._client = client
    
    @property
    def client(self) -> S3ClientProtocol:
        """Expose client for direct operations (e.g., head_bucket)."""
        return self._client
    
    def upload_file(
        self,
        local_path: Path | str,
        bucket_name: str,
        s3_key: str,
        is_verbose: bool = False,
    ) -> bool:
        """Upload a single file to S3/MinIO.
        
        Args:
            local_path: Local file path to upload.
            bucket_name: S3 bucket name.
            s3_key: S3 object key (path in bucket).
            is_verbose: Print detailed error messages.
            
        Returns:
            True if successful, False otherwise.
        """
        local_path = Path(local_path)
        
        valid, error = validate_local_file(local_path)
        if not valid:
            if is_verbose:
                print(f"✗ {error}")
            return False
        
        if not validate_bucket(self._client, bucket_name):
            if is_verbose:
                print(f"S3 Error: Cannot access bucket '{bucket_name}'")
            return False
        
        success, error = upload_file(self._client, local_path, bucket_name, s3_key)
        
        if not success and is_verbose:
            print(f"S3 upload error: {error}")
        
        return success
    
    def delete_prefix(
        self,
        bucket_name: str,
        prefix: str,
        is_verbose: bool = False,
    ) -> bool:
        """Delete all objects under an S3 prefix.
        
        Args:
            bucket_name: S3 bucket name.
            prefix: S3 prefix to delete.
            is_verbose: Print detailed messages.
            
        Returns:
            True if deletion successful, False otherwise.
        """
        if not validate_bucket(self._client, bucket_name):
            if is_verbose:
                print(f"S3 Error: Cannot access bucket '{bucket_name}'")
            return False
        
        deleted_count, error = delete_prefix(
            self._client, bucket_name, prefix
        )
        
        if error:
            if is_verbose:
                print(f"S3 delete error: {error}")
            return False
        
        if is_verbose and deleted_count > 0:
            print(f"✓ Deleted {deleted_count} objects from s3://{bucket_name}/{prefix}")
        
        return True

    def upload_directory(
        self,
        local_dir: Path | str,
        bucket_name: str,
        s3_prefix: Optional[str] = None,
        max_workers: int = 4,
        is_verbose: bool = False,
        clean_first: bool = False,
    ) -> bool:
        """Upload a directory with concurrent uploads.
        
        Args:
            local_dir: Local directory path to upload.
            bucket_name: S3 bucket name.
            s3_prefix: Optional prefix path in S3.
            max_workers: Number of concurrent upload threads.
            is_verbose: Print detailed error messages.
            clean_first: Delete existing objects under prefix before upload.
            
        Returns:
            True if all uploads successful, False otherwise.
        """
        local_dir = Path(local_dir)
        
        valid, error = validate_local_directory(local_dir)
        if not valid:
            if is_verbose:
                print(f"✗ {error}")
            return False
        
        if not validate_bucket(self._client, bucket_name):
            if is_verbose:
                print(f"S3 Error: Cannot access bucket '{bucket_name}'")
            return False
        
        # Clean existing objects if requested
        if clean_first and s3_prefix:
            self.delete_prefix(bucket_name, s3_prefix, is_verbose=False)
        
        tasks = collect_upload_tasks(local_dir, s3_prefix)
        
        if not tasks:
            if is_verbose:
                print(f"No files found in {local_dir}")
            return False
        
        results = upload_files_concurrently(
            self._client, tasks, bucket_name, max_workers
        )
        
        failed = [(f, e) for f, s, e in results if not s]
        
        if failed and is_verbose:
            print(f"S3 upload errors: {len(failed)} failed")
            for fname, err in failed[:3]:
                print(f"  - {fname}: {err}")
        
        return len(failed) == 0


@lru_cache(maxsize=1)
def get_s3_uploader() -> S3Uploader:
    """Get singleton S3 uploader instance.
    
    Returns:
        Cached S3Uploader instance with singleton client.
    """
    return S3Uploader(get_s3_client())


if __name__ == "__main__":
    pass

