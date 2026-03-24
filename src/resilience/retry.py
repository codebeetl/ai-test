"""Generic tenacity-based retry decorator for external service calls."""

import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient errors worth retrying."""
    retryable_messages = (
        "429", "quota", "rate limit", "resource exhausted",
        "503", "service unavailable", "timeout", "connection",
    )
    msg = str(exc).lower()
    return any(keyword in msg for keyword in retryable_messages)


def with_backoff(max_attempts: int = 5, min_wait: float = 2.0, max_wait: float = 60.0):
    """Return a tenacity retry decorator with exponential back-off.

    Retries only on transient/rate-limit errors. Fails fast on permanent
    errors (e.g. bad SQL, auth failures).
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=2, min=min_wait, max=max_wait),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
