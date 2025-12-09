from pathlib import Path


class Paths:
    """ Paths configuration """
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"
    ENV_FILE: Path = PROJECT_ROOT / ".env"


if __name__ == "__main__":
    pass