"""Structured logging setup using Loguru.

Call ``setup_logging()`` once at application startup (e.g. inside the FastAPI
lifespan or a CLI entrypoint).  All subsequent ``from loguru import logger``
calls will automatically use the configured handlers.

Standard library ``logging`` is also intercepted so that third-party libraries
that use it (e.g. uvicorn, httpx, sqlalchemy) are routed through Loguru.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from loguru import logger


class _InterceptHandler(logging.Handler):
    """Forward stdlib log records into Loguru.

    This handler bridges the standard ``logging`` module and Loguru so that
    any library that calls ``logging.getLogger(__name__).info(...)`` will be
    captured by Loguru's handlers.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Map stdlib level name to Loguru level
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk the call stack to find the originating frame (skip logging internals)
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(level: str = "INFO", log_dir: str = "logs") -> None:
    """Configure Loguru handlers and intercept the stdlib logging module.

    Args:
        level:    Minimum log level for the stderr handler (e.g. "DEBUG", "INFO").
        log_dir:  Directory where the rotating log file will be written.
    """
    # Remove Loguru's default stderr handler so we can add our own
    logger.remove()

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    # ── Console handler ────────────────────────────────────────────────────────
    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # ── Rotating file handler ──────────────────────────────────────────────────
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_path / "bcgx.log",
        format=log_format,
        level=level,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        backtrace=True,
        diagnose=False,  # Avoid leaking sensitive data in production log files
        enqueue=True,    # Thread-safe async writes
    )

    # ── Intercept stdlib logging ───────────────────────────────────────────────
    intercept_handler = _InterceptHandler()
    logging.basicConfig(handlers=[intercept_handler], level=0, force=True)

    # Ensure well-known noisy loggers are captured at the right level
    for noisy_logger in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logging.getLogger(noisy_logger).handlers = [intercept_handler]

    logger.info("Logging initialised — level={level}, log_dir={log_dir}", level=level, log_dir=str(log_path))


def get_logger(name: str) -> Any:
    """Return a Loguru logger bound with the given name.

    Prefer using ``from loguru import logger`` directly in modules.  This
    helper is provided for callers that want an explicitly named logger to
    mirror the stdlib ``logging.getLogger`` pattern.
    """
    return logger.bind(name=name)
