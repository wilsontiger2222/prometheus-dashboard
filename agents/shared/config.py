"""Configuration loader for Hivemind agents."""

import json
from pathlib import Path
from typing import Any


def load_agent_config(config_path: str, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load agent configuration from a JSON file, merging with optional defaults.

    Args:
        config_path: Path to the JSON configuration file.
        defaults: Optional dictionary of default values. File values override defaults.

    Returns:
        Merged configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        config = json.load(f)

    if defaults:
        merged = {**defaults, **config}
        return merged

    return config
