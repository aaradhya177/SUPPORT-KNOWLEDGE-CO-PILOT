"""Reusable logging utilities for the application."""

import json
import logging
from datetime import UTC, datetime
from typing import Final

from app.config import get_settings

STANDARD_LOG_RECORD_KEYS: Final[set[str]] = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


class JsonLogFormatter(logging.Formatter):
    """Format log records as compact JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Return a structured JSON log line."""
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key not in STANDARD_LOG_RECORD_KEYS and not key.startswith("_"):
                payload[key] = _json_safe(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


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
        handler.setFormatter(JsonLogFormatter())
        logger.addHandler(handler)

    for handler in logger.handlers:
        handler.setLevel(log_level)
        handler.setFormatter(JsonLogFormatter())

    return logger


def _json_safe(value: object) -> object:
    """Return a JSON-serializable value."""
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value
