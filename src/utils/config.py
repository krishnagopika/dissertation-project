"""
config.py — Config loading utility.
"""

from __future__ import annotations


def load_config(config_path: str) -> dict:
    """Load YAML config and return as dictionary.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed config as a dictionary.
    """
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
