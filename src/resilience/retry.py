"""Generic tenacity-based retry decorator for external service calls."""

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging

logger = logging.getLogger(__name__)


def with_backoff(max_attempts: int = 3, min_wait: int = 1, max_wait: int = 10):
    """Return a tenacity retry decorator with exponential back-off.

    Use this on any function that calls an external API (LLM, BigQuery)
    to make it resilient to transient failures and rate-limit errors.

    Args:
        max_attempts: Maximum number of total attempts.
        min_wait: Minimum seconds between retries.
        max_wait: Maximum seconds between retries.
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry attempt {retry_state.attempt_number}"
        ),
    )
