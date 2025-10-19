from enum import Enum
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

from configs.paths import ENV_FILE


class Environment(str, Enum):
      """ Environment enumeration """
      DEVELOPMENT = "development"
      STAGING = "staging"
      PRODUCTION = "production"

class Settings(BaseSettings):
      """ Application settings """
      environment: Environment = Environment.DEVELOPMENT

      model_config = ConfigDict(
            env_file_encoding="utf-8",
            env_file=str(ENV_FILE),
      )


if __name__ == "__main__":
      pass