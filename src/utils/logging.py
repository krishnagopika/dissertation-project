"""
logging.py — Shared logging setup.
"""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_dir: str, script_name: str) -> logging.Logger:
    """Set up logging to both file and console.

    Args:
        log_dir: Directory to write log file.
        script_name: Name used for the log file and logger.

    Returns:
        Configured logger instance.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    fh = logging.FileHandler(Path(log_dir) / f"{script_name}.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
