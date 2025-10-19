from pathlib import Path


# Project directory
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Directories
LOGS_DIR: Path = PROJECT_ROOT / "logs"
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"

ENV_FILE: Path = PROJECT_ROOT / ".env"


if __name__ == "__main__":
      pass