from enum import Enum
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Deployment(str, Enum):
    """Deployment environment."""
    DEV = 'development'
    STG = 'staging'
    PROD = 'production'

    @classmethod
    def _missing_(cls, value: object):
        """Handle alternative names for deployment environments."""
        if not isinstance(value, str):
            return None
        
        normalized = value.lower().strip()
        
        aliases: dict[str, 'Deployment'] = {
            # Development
            'dev': cls.DEV,
            'development': cls.DEV,
            'local': cls.DEV,
            # Staging
            'stg': cls.STG,
            'staging': cls.STG,
            'stage': cls.STG,
            'uat': cls.STG,
            # Production
            'prd': cls.PROD,
            'prod': cls.PROD,
            'production': cls.PROD,
            'live': cls.PROD,
        }
        
        return aliases.get(normalized)
    
    @property
    def is_prod(self) -> bool:
        """Check if this is a production environment."""
        return self == Deployment.PROD
    
    @property
    def is_dev(self) -> bool:
        """Check if this is a development environment."""
        return self == Deployment.DEV


class DeploymentSettings(BaseSettings):
    """Deployment configuration settings."""
    deployment: Deployment = Field(
        default=Deployment.DEV,
        alias='DEPLOYMENT',
        description="Deployment environment"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore'
    )


if __name__ == "__main__":
    pass