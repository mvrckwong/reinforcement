from enum import Enum
from typing import ClassVar

import pendulum
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
    """Deployment configuration settings with TTL-based auto-refresh."""
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
    
    # Class-level cache (excluded from Pydantic model)
    _instance: ClassVar['DeploymentSettings | None'] = None
    _loaded_at: ClassVar[pendulum.DateTime | None] = None
    _ttl: ClassVar[pendulum.Duration] = pendulum.duration(hours=1)
    
    @classmethod
    def get(cls, ttl: pendulum.Duration | None = None) -> 'DeploymentSettings':
        """
        Get settings instance with TTL-based auto-refresh.
        
        Args:
            ttl: Optional custom TTL. If provided, updates the default TTL.
        
        Returns:
            Cached or fresh DeploymentSettings instance.
        """
        if ttl is not None:
            cls._ttl = ttl
        
        now = pendulum.now()
        is_stale = cls._loaded_at is not None and (now - cls._loaded_at) > cls._ttl
        
        if cls._instance is None or is_stale:
            cls._instance = cls()
            cls._loaded_at = now
        
        return cls._instance
    
    @classmethod
    def refresh(cls) -> 'DeploymentSettings':
        """Force reload settings from .env file, ignoring TTL."""
        cls._instance = cls()
        cls._loaded_at = pendulum.now()
        return cls._instance


if __name__ == "__main__":
    pass
