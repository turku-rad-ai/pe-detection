from typing import Any

import yaml


def config(config_file: str = "config.yml") -> Any:
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
