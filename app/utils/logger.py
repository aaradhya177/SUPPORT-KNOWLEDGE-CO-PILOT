"""Reusable logging utilities for the application."""

import logging
from typing import Final

from app.config import get_settings

LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger instance.

    Args:
        name: Logger name, usually ``__name__`` from the caller.

    Returns:
        A standard library logger configured with a structured text formatter.
    """
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(handler)

    for handler in logger.handlers:
        handler.setLevel(log_level)

    return logger
