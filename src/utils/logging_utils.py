"""Logging utility functions."""

from __future__ import annotations

import logging


def setup_logger(name: str = "nq_research", level: int = logging.INFO) -> logging.Logger:
    """Create and configure a simple stream logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
