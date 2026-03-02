from pathlib import Path

import yaml
from utils.logger import logger


def load_config(path: str):
    config_path = Path(path)

    if not config_path.exists():
        logger.error(f"Error occured due to file not found")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
