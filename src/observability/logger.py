"""Centralised structured logging setup for the retail data agent.

All logs are written to data/agent.log as newline-delimited JSON.
Monitor in a separate terminal with:
    tail -f data/agent.log | python -m json.tool
"""

import logging
import json
from pathlib import Path
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

_SKIP_ATTRS = frozenset(logging.LogRecord.__dict__.keys()) | {
    "message", "asctime", "args", "exc_info", "exc_text",
    "stack_info", "taskName",
}

LOG_PATH = Path(__file__).parent.parent.parent / "data" / "agent.log"


class _SafeEncoder(json.JSONEncoder):
    """JSON encoder that coerces any non-serialisable value to its string repr."""
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON for structured log ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, val in record.__dict__.items():
            if key not in _SKIP_ATTRS and not key.startswith("_"):
                try:
                    json.dumps(val)
                    log_obj[key] = val
                except (TypeError, ValueError):
                    log_obj[key] = str(val)
        return json.dumps(log_obj, cls=_SafeEncoder)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging to write JSON to a rotating file, keeping stdout clean.

    Log file: data/agent.log (max 5 MB, 3 backups)
    Monitor:  tail -f data/agent.log

    Args:
        level: Logging level (default INFO).
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Rotating file handler — JSON structured logs
    file_handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(JsonFormatter())
    file_handler.setLevel(level)

    # Minimal stdout handler — WARNING and above only, plain text
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter("⚠️  %(levelname)s: %(message)s"))
    stdout_handler.setLevel(logging.WARNING)

    logging.basicConfig(level=level, handlers=[file_handler, stdout_handler])

    # Suppress noisy third-party loggers from the console (still written to file)
    for noisy in (
        "httpx", "httpcore", "google.auth", "urllib3",
        "tenacity",                    # suppresses raw retry WARNING lines
        "langchain_google_genai",      # suppresses ChatGoogleGenerativeAIError dumps
        "langchain_core",
        "google.api_core",
        "google.generativeai",
    ):
        noisy_logger = logging.getLogger(noisy)
        noisy_logger.setLevel(logging.WARNING)
        # Remove any existing stdout handlers these libraries may have added
        noisy_logger.propagate = False
        noisy_logger.addHandler(file_handler)  # keep writing to file
