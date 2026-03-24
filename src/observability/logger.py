"""Centralised structured logging setup for the retail data agent.

Call setup_logging() once in main.py before the agent runs.
All modules use logging.getLogger(__name__) — this module configures
the root handler to emit JSON-structured lines for easy ingestion by
log aggregators (Datadog, GCP Logging, etc.).
"""

import logging
import json
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON for structured log ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise a LogRecord to JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON string containing timestamp, level, logger, message, and extras.
        """
        log_obj = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach any extra fields passed via extra={} in log calls
        for key, val in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                log_obj[key] = val
        return json.dumps(log_obj)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with JSON formatter.

    Args:
        level: Logging level (default INFO).
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler])
