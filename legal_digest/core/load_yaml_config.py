from pathlib import Path

import yaml


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


config = load_config("config/training.yaml")
