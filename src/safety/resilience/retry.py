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

# Daily quota exhaustion messages — retrying won't help, fail fast
_PERMANENT_QUOTA_PHRASES = (
    "generaterequestsperday",   # GenerateRequestsPerDayPerProjectPerModel
    "exceeded your current quota",
    "check your plan and billing",
)


def _is_retryable(exc: BaseException) -> bool:
    """Return True only for transient errors that are worth retrying.

    Distinguishes:
      - Transient (retry): per-minute rate limits, 503, timeout, connection errors
      - Permanent (fail fast): daily quota exhaustion, auth errors, bad requests

    Daily quota errors (429 + 'exceeded your current quota') are NOT retried
    because waiting 60s won't help — the quota resets at midnight UTC.
    """
    msg = str(exc).lower()

    # Fail fast on permanent quota exhaustion
    if any(phrase in msg for phrase in _PERMANENT_QUOTA_PHRASES):
        return False

    retryable_messages = (
        "429", "quota", "rate limit", "resource exhausted",
        "503", "service unavailable", "timeout", "connection",
    )
    return any(keyword in msg for keyword in retryable_messages)


def with_backoff(max_attempts: int = 5, min_wait: float = 2.0, max_wait: float = 60.0):
    """Return a tenacity retry decorator with exponential back-off.

    Retries only on transient/rate-limit errors. Fails fast on permanent
    errors (daily quota exhaustion, auth failures, bad requests).

    Args:
        max_attempts: Maximum total attempts (default 5).
        min_wait: Minimum seconds between retries (default 2).
        max_wait: Maximum seconds between retries (default 60).
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=2, min=min_wait, max=max_wait),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
