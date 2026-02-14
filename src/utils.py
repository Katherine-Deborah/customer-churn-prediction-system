"""
utils.py — Shared utility functions for ChurnSight.

Provides:
  - A consistent logger for every module
  - A helper to ensure output directories exist
"""

import logging
import os


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a standard format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(name)-14s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dirs(*dirs: str) -> None:
    """Create directories if they don't already exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
