"""Logging utilities for the bot application."""
from __future__ import annotations

import logging
from logging import Logger
from typing import Optional

_LOGGER_NAME = "bot"


def setup_logging(level: int = logging.INFO, log_format: Optional[str] = None) -> Logger:
    """Configure application-wide logging and return the configured logger.

    TODO: Extend logging to support structured logging and external sinks.
    """

    formatter = logging.Formatter(
        fmt=log_format
        or "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False

    logger.debug("Logging has been configured.")
    return logger
