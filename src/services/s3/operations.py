"""
Pure S3 upload and delete operations.

Usage:
    from services.s3 import get_s3_client, upload_file, delete_prefix
    
    client = get_s3_client()
    success = upload_file(client, path, bucket, key)
    delete_prefix(client, bucket, "model/run_id/checkpoints/latest")
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from services.s3.client import S3ClientProtocol

if TYPE_CHECKING:
    from botocore.client import BaseClient


def delete_prefix(
    client: "BaseClient",
    bucket_name: str,
    prefix: str,
) -> tuple[int, str]:
    """Delete all objects under an S3 prefix.
    
    Args:
        client: S3 client instance (needs list/delete permissions).
        bucket_name: S3 bucket name.
        prefix: S3 prefix to delete (e.g., 'model/run_id/checkpoints/latest').
        
    Returns:
        Tuple of (deleted_count, error_message).
    """
    try:
        # List all objects under prefix
        paginator = client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        
        deleted_count = 0
        for page in pages:
            objects = page.get('Contents', [])
            if not objects:
                continue
            
            # Delete in batches of 1000 (S3 limit)
            delete_keys = [{'Key': obj['Key']} for obj in objects]
            client.delete_objects(
                Bucket=bucket_name,
                Delete={'Objects': delete_keys}
            )
            deleted_count += len(delete_keys)
        
        return (deleted_count, "")
    except Exception as e:
        return (0, str(e))


def validate_file(path: Path) -> bool:
    """Validate path exists and is a file."""
    return path.exists() and path.is_file()


def validate_dir(path: Path) -> bool:
    """Validate path exists and is a directory."""
    return path.exists() and path.is_dir()


def collect_upload_tasks(
    local_dir: Path,
    s3_prefix: str | None = None,
) -> list[tuple[Path, str]]:
    """Collect files with their S3 keys.
    
    Args:
        local_dir: Local directory to traverse.
        s3_prefix: Optional prefix for S3 keys.
        
    Returns:
        List of (local_path, s3_key) tuples.
    """
    tasks = []
    for file_path in local_dir.rglob('*'):
        if file_path.is_file():
            relative = file_path.relative_to(local_dir)
            key = f"{s3_prefix}/{relative}" if s3_prefix else str(relative)
            tasks.append((file_path, key.replace('\\', '/')))
    return tasks


def upload_file(
    client: S3ClientProtocol,
    local_path: Path,
    bucket_name: str,
    s3_key: str,
) -> bool:
    """Upload single file. Returns True on success, False on failure."""
    try:
        client.upload_file(str(local_path), bucket_name, s3_key)
        return True
    except Exception:
        return False


def upload_files_concurrently(
    client: S3ClientProtocol,
    tasks: list[tuple[Path, str]],
    bucket_name: str,
    max_workers: int = 4,
) -> list[tuple[str, bool]]:
    """Upload multiple files concurrently.
    
    Args:
        client: S3 client instance.
        tasks: List of (local_path, s3_key) tuples.
        bucket_name: S3 bucket name.
        max_workers: Number of concurrent upload threads.
        
    Returns:
        List of (filename, success) tuples.
    """
    results: list[tuple[str, bool]] = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(upload_file, client, path, bucket_name, key): path.name
            for path, key in tasks
        }
        for future in as_completed(futures):
            filename = futures[future]
            success = future.result()
            results.append((filename, success))
    
    return results


if __name__ == "__main__":
    pass

