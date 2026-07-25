"""structlog setup: one JSON line per stage transition, to stderr and a file."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any

import structlog

# Telegram puts the bot token in the URL path, and httpx logs full URLs at INFO.
# Left alone that writes the token in plaintext to the JSONL log on every run.
# Two defences: silence the HTTP client loggers, and scrub anything that still
# looks like a secret on its way out.
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "yt_dlp")

_REDACTIONS = (
    re.compile(r"(api\.telegram\.org/bot)[^/\s]+"),
    re.compile(r"(sk-[A-Za-z0-9]{4})[A-Za-z0-9]{8,}"),
    re.compile(r"((?:token|api[_-]?key|secret|password)[\"']?\s*[:=]\s*[\"']?)[^\s\"',}]+", re.I),
)


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        for pattern in _REDACTIONS:
            value = pattern.sub(r"\1<redacted>", value)
        return value
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_redact(v) for v in value)
    return value


def _redact_processor(_logger: Any, _name: str, event_dict: dict) -> dict:
    return _redact(event_dict)


def setup_logging(log_path: Path | None = None, level: str = "INFO") -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        format="%(message)s",
        handlers=handlers,
        level=getattr(logging, level.upper(), logging.INFO),
        force=True,
    )

    # Never let a third-party client log a URL containing credentials.
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _redact_processor,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "ytdigest") -> structlog.BoundLogger:
    return structlog.get_logger(name)
