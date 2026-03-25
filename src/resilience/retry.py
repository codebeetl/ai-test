"""Generic tenacity-based retry decorator for external service calls.

Retry warnings from tenacity are intentionally suppressed from the console.
The quota_guard module converts exhausted-quota errors into user-friendly
messages; transient retries show a clean "Retrying..." progress indicator.
"""

import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    RetryCallState,
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


def _quiet_before_sleep(retry_state: RetryCallState) -> None:
    """Silent tenacity before_sleep hook — logs at DEBUG only, shows clean progress.

    Replaces before_sleep_log() which would print the full raw exception
    as a WARNING to the console on every retry attempt.
    """
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    wait = getattr(retry_state.next_action, "sleep", None)
    attempt = retry_state.attempt_number

    # Log full detail to file (DEBUG — won't appear on console)
    logger.debug(
        "Retrying after transient error",
        extra={
            "attempt": attempt,
            "wait_s": round(wait, 1) if wait else None,
            "error": str(exc)[:300] if exc else None,
        },
    )

    # Show a clean single-line progress update instead of the raw traceback
    try:
        from src.observability.progress import show as _progress
        wait_str = f"{round(wait)}s" if wait else "..."
        _progress(f"Rate limited — retrying in {wait_str} (attempt {attempt})...")
    except Exception:
        pass  # never let progress display failure break the retry loop


def with_backoff(max_attempts: int = 5, min_wait: float = 2.0, max_wait: float = 60.0):
    """Return a tenacity retry decorator with exponential back-off.

    Retries only on transient/rate-limit errors. Fails fast on permanent
    errors (e.g. bad SQL, auth failures).  Console output on retry is a
    clean progress line, not a raw exception dump.
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=2, min=min_wait, max=max_wait),
        retry=retry_if_exception(_is_retryable),
        before_sleep=_quiet_before_sleep,
        reraise=True,
    )
