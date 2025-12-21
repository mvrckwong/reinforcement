"""
S3 upload service.

Usage:
    from services.s3 import get_s3_uploader
    
    uploader = get_s3_uploader()
    uploader.upload_file(local_path, bucket, s3_key)
    uploader.upload_directory(local_dir, bucket, s3_prefix)
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Optional

from services.s3.client import (
    S3ClientProtocol,
    validate_bucket,
    get_s3_client,
    delete_prefix,
    upload_file,
)


class S3Uploader:
    """Orchestrator for S3 upload operations with validation and logging."""
    
    def __init__(self, client: S3ClientProtocol):
        """Initialize with injected S3 client (DIP)."""
        self._client = client
    
    @property
    def client(self) -> S3ClientProtocol:
        """Expose client for direct operations (e.g., lifecycle rules)."""
        return self._client
    
    def upload_file(
        self,
        local_path: Path | str,
        bucket_name: str,
        s3_key: str,
        is_verbose: bool = False,
    ) -> bool:
        """Upload a single file to S3/MinIO with validation."""
        local_path = Path(local_path)
        
        if not (local_path.exists() and local_path.is_file()):
            if is_verbose:
                print(f"✗ Invalid file path: {local_path}")
            return False
        
        if not validate_bucket(self._client, bucket_name):
            if is_verbose:
                print(f"S3 Error: Cannot access bucket '{bucket_name}'")
            return False
        
        success = upload_file(self._client, local_path, bucket_name, s3_key)
        
        if not success and is_verbose:
            print(f"✗ S3 upload failed: {s3_key}")
        
        return success
    
    def delete_prefix(
        self,
        bucket_name: str,
        prefix: str,
        is_verbose: bool = False,
    ) -> bool:
        """Delete all objects under an S3 prefix."""
        if not validate_bucket(self._client, bucket_name):
            if is_verbose:
                print(f"S3 Error: Cannot access bucket '{bucket_name}'")
            return False
        
        deleted_count, error = delete_prefix(self._client, bucket_name, prefix)
        
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
        """Upload a directory with concurrent uploads."""
        local_dir = Path(local_dir)
        
        if not (local_dir.exists() and local_dir.is_dir()):
            if is_verbose:
                print(f"✗ Invalid directory path: {local_dir}")
            return False
        
        if not validate_bucket(self._client, bucket_name):
            if is_verbose:
                print(f"S3 Error: Cannot access bucket '{bucket_name}'")
            return False
        
        if clean_first and s3_prefix:
            self.delete_prefix(bucket_name, s3_prefix, is_verbose=False)
        
        # Collect upload tasks (file path → S3 key)
        tasks = []
        for file_path in local_dir.rglob('*'):
            if file_path.is_file():
                relative = file_path.relative_to(local_dir)
                key = f"{s3_prefix}/{relative}" if s3_prefix else str(relative)
                tasks.append((file_path, key.replace('\\', '/')))
        
        if not tasks:
            if is_verbose:
                print(f"No files found in {local_dir}")
            return False
        
        # Concurrent uploads
        failed = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    upload_file, self._client, path, bucket_name, key
                ): path.name
                for path, key in tasks
            }
            for future in as_completed(futures):
                if not future.result():
                    failed.append(futures[future])
        
        if failed and is_verbose:
            print(f"S3 upload errors: {len(failed)} failed")
            for fname in failed[:3]:
                print(f"  - {fname}")
        
        return len(failed) == 0


@lru_cache(maxsize=1)
def get_s3_uploader() -> S3Uploader:
    """Get singleton S3 uploader instance."""
    return S3Uploader(get_s3_client())


if __name__ == "__main__":
    pass