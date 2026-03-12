"""
Centralised logging configuration.
"""

import logging
from typing import Optional


def setup_logger(name: str = "visionframework", level: int = logging.INFO,
                 fmt: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            fmt or "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def get_logger(name: str = "visionframework") -> logging.Logger:
    return logging.getLogger(name)
