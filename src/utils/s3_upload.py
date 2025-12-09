from boto3 import client as boto3_client
from pathlib import Path
from typing import Optional
from botocore.client import Config, BaseClient
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed

from configs.storage import S3Settings


class S3Uploader:
    """Handles S3/MinIO upload operations with dependency injection and reusable client."""
    
    def __init__(self, settings: Optional[S3Settings] = None):
        """
        Initialize the S3 uploader.
        
        Args:
            settings: S3 settings to use. If None, will use default S3Settings()
        """
        self._settings = settings or S3Settings()
        self._client: Optional[BaseClient] = None
    
    @staticmethod
    def _upload_single_file(
        s3_client: BaseClient, 
        file_path: Path, 
        bucket_name: str, 
        s3_key: str
    ) -> tuple[str, bool, str]:
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
    
    @property
    def client(self) -> BaseClient:
        """Lazy-load S3 client (creates only when first accessed)."""
        if self._client is None:
            self._client = self._create_client()
        return self._client
    
    def _create_client(self) -> BaseClient:
        """Create and return an S3 client configured for MinIO with retry logic."""
        return boto3_client(
            's3',
            endpoint_url=self._settings.endpoint_url,
            aws_access_key_id=self._settings.access_key_id,
            aws_secret_access_key=self._settings.secret_access_key,
            region_name=self._settings.region,
            config=Config(
                signature_version='s3v4',
                retries={
                    'max_attempts': 3,
                    'mode': 'adaptive'  # Adapts retry strategy based on response
                },
                connect_timeout=5,
                read_timeout=60
            ),
            use_ssl=self._settings.use_ssl
        )
    
    def _validate_bucket(self, bucket_name: str, verbose: bool = False) -> bool:
        """Validate that the bucket exists and is accessible."""
        try:
            self.client.head_bucket(Bucket=bucket_name)
            return True
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
    
    def _collect_files(
        self, 
        local_dir: Path, 
        s3_prefix: Optional[str] = None
    ) -> list[tuple[Path, str]]:
        """Collect all files to upload with their S3 keys."""
        files_to_upload = []
        
        for file_path in local_dir.rglob('*'):
            if file_path.is_file():
                # Calculate relative path from local_dir
                relative_path = file_path.relative_to(local_dir)
                
                # Construct S3 key
                if s3_prefix:
                    s3_key = f"{s3_prefix}/{relative_path}".replace('\\', '/')
                else:
                    s3_key = str(relative_path).replace('\\', '/')
                
                files_to_upload.append((file_path, s3_key))
        
        return files_to_upload
    
    def _upload_files_concurrently(
        self,
        files_to_upload: list[tuple[Path, str]],
        bucket_name: str,
        max_workers: int,
        verbose: bool
    ) -> bool:
        """Upload files concurrently and handle results."""
        successful = 0
        failed_files = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all upload tasks
            futures = {
                executor.submit(
                    self._upload_single_file, 
                    self.client, 
                    file_path, 
                    bucket_name, 
                    s3_key
                ): file_path.name
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
    
    def upload_directory(
        self,
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
            local_dir = Path(local_dir)
            
            if not local_dir.exists():
                if verbose:
                    print(f"✗ Error: Directory does not exist: {local_dir}")
                return False
            
            # Validate bucket exists
            if not self._validate_bucket(bucket_name, verbose):
                return False
            
            # Collect all files to upload
            files_to_upload = self._collect_files(local_dir, s3_prefix)
            
            if not files_to_upload:
                if verbose:
                    print(f"No files found in {local_dir}")
                return False
            
            # Upload files concurrently
            return self._upload_files_concurrently(
                files_to_upload, 
                bucket_name, 
                max_workers, 
                verbose
            )
            
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