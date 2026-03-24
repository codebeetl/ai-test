"""Tenacity-based retry decorator for external API / 3rd-party service calls.

Provides a pre-configured decorator for wrapping calls that may fail due to
transient network issues, rate limits, or API downtime. Used by:
  - BigQueryRunner.execute_query
  - Gemini LLM invocations
  - ChromaDB vector store queries

Backoff strategy:
  Exponential backoff with jitter prevents thundering-herd on shared APIs.
  Max 3 attempts with a 60-second ceiling prevents excessive wait times in
  interactive CLI sessions.
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Callable, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)

MAX_ATTEMPTS = 3
MIN_WAIT_SECONDS = 1
MAX_WAIT_SECONDS = 60


def resilient(
    *,
    exception_types: tuple[type[Exception], ...] = (Exception,),
    max_attempts: int = MAX_ATTEMPTS,
) -> Callable[[F], F]:
    """Decorator factory: wraps a callable with exponential-backoff retry logic.

    Usage::

        @resilient(exception_types=(GoogleCloudError, TimeoutError))
        def call_bigquery(sql: str) -> pd.DataFrame:
            ...

    Args:
        exception_types: Tuple of exception classes that should trigger a retry.
            Defaults to all Exception subclasses.
        max_attempts: Total number of attempts including the first try.

    Returns:
        Decorated function that retries on the specified exceptions.
    """

    def decorator(func: F) -> F:
        @retry(
            retry=retry_if_exception_type(exception_types),
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential_jitter(
                initial=MIN_WAIT_SECONDS, max=MAX_WAIT_SECONDS
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
