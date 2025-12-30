from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class S3Settings(BaseSettings):
    """S3/MinIO configuration settings."""
    
    endpoint_url: str = Field(
        ..., 
        alias='S3_ENDPOINT_URL', 
        description="S3/MinIO endpoint URL (e.g., https://localhost:9000)"
    )
    access_key_id: str = Field(
        ..., 
        alias='S3_ACCESS_KEY_ID', 
        description="S3/MinIO access key ID"
    )
    secret_access_key: str = Field(
        ..., 
        alias='S3_SECRET_ACCESS_KEY', 
        description="S3/MinIO secret access key"
    )
    region: str = Field(
        default='local', 
        alias='S3_REGION', 
        description="S3/MinIO region (e.g., us-east-1)"
    )

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


if __name__ == "__main__":
    pass