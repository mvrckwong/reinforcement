from boto3 import client as boto3_client
from pathlib import Path
from typing import Optional
from botocore.client import Config, BaseClient
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class S3Settings(BaseSettings):
    """S3/MinIO configuration settings loaded from environment variables."""
    
    endpoint_url: str = Field(..., alias='S3_ENDPOINT_URL', description="S3/MinIO endpoint URL (e.g., https://localhost:9000)")
    access_key_id: str = Field(..., alias='S3_ACCESS_KEY_ID', description="S3 access key ID")
    secret_access_key: str = Field(..., alias='S3_SECRET_ACCESS_KEY', description="S3 secret access key")
    region: str = Field(default='us-east-1', alias='S3_REGION', description="AWS region")

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore'
    )
    
    @property
    def use_ssl(self) -> bool:
        """Determine if SSL should be used based on endpoint URL."""
        return self.endpoint_url.startswith('https')


def get_s3_client() -> BaseClient:
    """Create and return an S3 client configured for MinIO with retry logic."""
    settings = S3Settings()
    
    return boto3_client(
        's3',
        endpoint_url=settings.endpoint_url,
        aws_access_key_id=settings.access_key_id,
        aws_secret_access_key=settings.secret_access_key,
        region_name=settings.region,
        config=Config(
            signature_version='s3v4',
            retries={
                'max_attempts': 3,
                'mode': 'adaptive'  # Adapts retry strategy based on response
            },
            connect_timeout=5,
            read_timeout=60
        ),
        use_ssl=settings.use_ssl
    )


def _upload_single_file(s3_client, file_path: Path, bucket_name: str, s3_key: str) -> tuple[str, bool, str]:
    """Upload a single file to S3 (retries handled by boto3 client config)."""
    try:
        s3_client.upload_file(
            str(file_path),
            bucket_name,
            s3_key
        )
        return (file_path.name, True, "")
    except Exception as e:
        return (file_path.name, False, str(e))


def upload_directory_to_s3(
    local_dir: Path | str,
    bucket_name: str,
    s3_prefix: Optional[str] = None,
    max_workers: int = 4,
    verbose: bool = False,
) -> bool:
    """
    Upload a directory and all its contents to S3/MinIO with concurrent uploads.
    
    Args:
        local_dir: Local directory path to upload
        bucket_name: S3 bucket name
        s3_prefix: Optional prefix path in S3 (e.g., 'impala_cartpole/20241207_120000')
        max_workers: Number of concurrent upload threads (default: 4)
        verbose: Print detailed error messages (default: False)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        s3_client = get_s3_client()
        local_dir = Path(local_dir)
        
        if not local_dir.exists():
            print(f"✗ Error: Directory does not exist: {local_dir}")
            return False
        
        # Validate bucket exists (silently)
        try:
            s3_client.head_bucket(Bucket=bucket_name)
        except ClientError as e:
            if verbose:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code == '404':
                    print(f"S3 Error: Bucket '{bucket_name}' does not exist")
                elif error_code == '403':
                    print(f"S3 Error: Access denied to bucket '{bucket_name}'")
                else:
                    print(f"S3 Error: {e}")
            return False
        
        # Collect all files to upload
        files_to_upload = []
        for file_path in local_dir.rglob('*'):
            if file_path.is_file():
                # Calculate relative path from local_dir (not local_dir.parent)
                relative_path = file_path.relative_to(local_dir)
                
                # Construct S3 key
                if s3_prefix:
                    s3_key = f"{s3_prefix}/{relative_path}".replace('\\', '/')
                else:
                    s3_key = str(relative_path).replace('\\', '/')
                
                files_to_upload.append((file_path, s3_key))
        
        if not files_to_upload:
            if verbose:
                print(f"No files found in {local_dir}")
            return False
        
        # Upload files concurrently
        successful = 0
        failed_files = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all upload tasks
            futures = {
                executor.submit(_upload_single_file, s3_client, file_path, bucket_name, s3_key): file_path.name
                for file_path, s3_key in files_to_upload
            }
            
            # Process completed uploads
            for future in as_completed(futures):
                filename, success, error = future.result()
                if success:
                    successful += 1
                else:
                    failed_files.append((filename, error))
        
        if failed_files:
            if verbose:
                print(f"S3 upload errors: {len(failed_files)} failed")
                for fname, err in failed_files[:3]:  # Show first 3 errors
                    print(f"  - {fname}: {err}")
            return False
        
        return True
        
    except ValueError as e:
        if verbose:
            print(f"S3 config error: {e}")
        return False
    except ClientError as e:
        if verbose:
            error_code = e.response.get('Error', {}).get('Code', '')
            error_msg = e.response.get('Error', {}).get('Message', str(e))
            
            if 'InvalidArgument' in error_code and 'API port' in error_msg:
                print(f"S3 Error: Wrong API port (use 9000, not 9002)")
            else:
                print(f"S3 Error ({error_code}): {error_msg}")
        return False
    except Exception as e:
        if verbose:
            print(f"S3 upload error: {e}")
        return False


if __name__ == "__main__":
    pass