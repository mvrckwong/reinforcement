import os
import boto3
from pathlib import Path
from typing import Optional
from botocore.client import Config


def get_s3_client():
    """Create and return an S3 client configured for MinIO."""
    endpoint_url = os.getenv('S3_ENDPOINT_URL')
    if not endpoint_url:
        raise ValueError("S3_ENDPOINT_URL not set in environment")
    
    access_key = os.getenv('S3_ACCESS_KEY_ID')
    secret_key = os.getenv('S3_SECRET_ACCESS_KEY')
    
    if not access_key or not secret_key:
        raise ValueError("S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY must be set")
    
    return boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=os.getenv('S3_REGION', 'us-east-1'),
        config=Config(signature_version='s3v4'),
        use_ssl=False  # Set to True if using https://
    )


def upload_directory_to_s3(
    local_dir: Path,
    bucket_name: str,
    s3_prefix: Optional[str] = None,
) -> bool:
    """
    Upload a directory and all its contents to S3/MinIO.
    
    Args:
        local_dir: Local directory path to upload
        bucket_name: S3 bucket name
        s3_prefix: Optional prefix path in S3 (e.g., 'checkpoints/impala')
    
    Returns:
        True if successful, False otherwise
    """
    try:
        s3_client = get_s3_client()
        local_dir = Path(local_dir)
        
        file_count = 0
        # Walk through all files in the directory
        for file_path in local_dir.rglob('*'):
            if file_path.is_file():
                # Calculate relative path for S3 key
                relative_path = file_path.relative_to(local_dir.parent)
                
                # Construct S3 key
                if s3_prefix:
                    s3_key = f"{s3_prefix}/{relative_path}".replace('\\', '/')
                else:
                    s3_key = str(relative_path).replace('\\', '/')
                
                # Upload file
                print(f"  Uploading {file_path.name} to s3://{bucket_name}/{s3_key}")
                s3_client.upload_file(
                    str(file_path),
                    bucket_name,
                    s3_key
                )
                file_count += 1
        
        print(f"✓ Successfully uploaded {file_count} files from {local_dir.name} to S3")
        return True
        
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        print("  Please check your .env file settings")
        return False
    except Exception as e:
        error_msg = str(e)
        if "API port" in error_msg:
            print(f"✗ Error: Wrong port for MinIO API")
            print(f"  Hint: Port 9002 is typically the console UI. Try port 9000 for the API.")
            print(f"  Update S3_ENDPOINT_URL in your .env file to: http://192.168.1.105:9000")
        else:
            print(f"✗ Error uploading to S3: {e}")
        return False


if __name__ == "__main__":
    pass