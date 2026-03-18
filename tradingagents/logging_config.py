"""Logging configuration for TradingAgents framework."""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """Configure logging for the TradingAgents framework.

    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string. If None, uses default.
        log_file: Optional file path to write logs to.
    """
    if format_string is None:
        format_string = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"

    handlers = [logging.StreamHandler(sys.stderr)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
    )

    # Set third-party loggers to WARNING to reduce noise
    for noisy_logger in ("httpx", "httpcore", "urllib3", "openai", "anthropic"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name, prefixed with 'tradingagents'.

    Args:
        name: Logger name (will be prefixed with 'tradingagents.')

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(f"tradingagents.{name}")
