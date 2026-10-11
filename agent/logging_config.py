"""
Centralised logging setup for ClinicalMind.

Call setup_logging() once at application startup (ui/app.py, __main__ blocks).
Every module then does only:

    import structlog
    log = structlog.get_logger()

How it works
------------
structlog is configured to use the stdlib logging module as its backend.
stdlib logging is configured with two handlers:

  1. StreamHandler (stderr) — coloured, human-readable for the terminal
  2. RotatingFileHandler  — JSON lines in logs/clinicalmind.log,
                            10 MB per file, 7 backups kept

This means every structlog log call (log.info / log.warning / log.error)
automatically goes to BOTH destinations.

Log file location:  <project-root>/logs/clinicalmind.log
Log level:          INFO by default; override with LOG_LEVEL env var
                    (e.g. LOG_LEVEL=DEBUG python3 -m agent.graph)
"""

import logging
import logging.handlers
import os
from pathlib import Path

import structlog

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT     = Path(__file__).parent.parent
_LOG_DIR  = _ROOT / "logs"
_LOG_FILE = _LOG_DIR / "clinicalmind.log"

_LOG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_configured = False   # guard: only configure once per process


def setup_logging(level: str | None = None) -> None:
    """
    Configure structlog + stdlib logging to write to console and log file.

    Safe to call multiple times — subsequent calls are no-ops.

    Parameters
    ----------
    level : str, optional
        Override log level, e.g. "DEBUG". Defaults to the LOG_LEVEL env var,
        or "INFO" if that is not set.
    """
    global _configured
    if _configured:
        return
    _configured = True

    # ------------------------------------------------------------------ #
    # 1. Resolve log level
    # ------------------------------------------------------------------ #
    level_name  = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    numeric_lvl = getattr(logging, level_name, logging.INFO)

    # ------------------------------------------------------------------ #
    # 2. Shared processors — run on every log record before rendering
    # ------------------------------------------------------------------ #
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
    ]

    # ------------------------------------------------------------------ #
    # 3. stdlib logging — handlers
    # ------------------------------------------------------------------ #

    # Console: human-readable coloured output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor = structlog.dev.ConsoleRenderer(colors=True),
            foreign_pre_chain = shared_processors,
        )
    )

    # File: newline-delimited JSON — easy to grep / ship to a log aggregator
    file_handler = logging.handlers.RotatingFileHandler(
        filename    = _LOG_FILE,
        maxBytes    = 10 * 1024 * 1024,   # 10 MB
        backupCount = 7,
        encoding    = "utf-8",
    )
    file_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor = structlog.processors.JSONRenderer(),
            foreign_pre_chain = shared_processors,
        )
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(numeric_lvl)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "langsmith", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ------------------------------------------------------------------ #
    # 4. structlog — route through stdlib, use shared processors
    # ------------------------------------------------------------------ #
    structlog.configure(
        processors = shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory       = structlog.stdlib.LoggerFactory(),
        wrapper_class        = structlog.make_filtering_bound_logger(numeric_lvl),
        cache_logger_on_first_use = True,
    )
