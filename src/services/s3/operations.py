"""
Pure S3 upload operations.

Usage:
    from services.s3 import get_s3_client, upload_file, collect_upload_tasks
    
    client = get_s3_client()
    success, error = upload_file(client, path, bucket, key)
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from services.s3.client import S3ClientProtocol


def validate_local_file(path: Path) -> tuple[bool, str]:
    """Validate local file exists.
    
    Args:
        path: Path to validate.
        
    Returns:
        Tuple of (valid, error_message).
    """
    if not path.exists():
        return (False, f"File does not exist: {path}")
    if not path.is_file():
        return (False, f"Path is not a file: {path}")
    return (True, "")


def validate_local_directory(path: Path) -> tuple[bool, str]:
    """Validate local directory exists.
    
    Args:
        path: Path to validate.
        
    Returns:
        Tuple of (valid, error_message).
    """
    if not path.exists():
        return (False, f"Directory does not exist: {path}")
    if not path.is_dir():
        return (False, f"Path is not a directory: {path}")
    return (True, "")


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
) -> tuple[bool, str]:
    """Upload single file.
    
    Args:
        client: S3 client instance.
        local_path: Local file path.
        bucket_name: S3 bucket name.
        s3_key: S3 object key.
        
    Returns:
        Tuple of (success, error_message).
    """
    try:
        client.upload_file(str(local_path), bucket_name, s3_key)
        return (True, "")
    except Exception as e:
        return (False, str(e))


def upload_files_concurrently(
    client: S3ClientProtocol,
    tasks: list[tuple[Path, str]],
    bucket_name: str,
    max_workers: int = 4,
) -> list[tuple[str, bool, str]]:
    """Upload multiple files concurrently.
    
    Args:
        client: S3 client instance.
        tasks: List of (local_path, s3_key) tuples.
        bucket_name: S3 bucket name.
        max_workers: Number of concurrent upload threads.
        
    Returns:
        List of (filename, success, error_message) tuples.
    """
    results: list[tuple[str, bool, str]] = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(upload_file, client, path, bucket_name, key): path.name
            for path, key in tasks
        }
        for future in as_completed(futures):
            filename = futures[future]
            success, error = future.result()
            results.append((filename, success, error))
    
    return results


if __name__ == "__main__":
    pass

