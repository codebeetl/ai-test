"""Quota and rate-limit guard for LLM and external API calls.

Catches exhausted-quota / daily-limit errors that tenacity will NOT
successfully retry (because they won't recover within the session) and
converts them into user-friendly messages, while logging the full detail
for operator investigation.

Usage:
    from src.resilience.quota_guard import quota_safe_invoke

    response = quota_safe_invoke(_invoke_llm, prompt_messages)
    if isinstance(response, dict) and response.get("quota_error"):
        # surface response["message"] to the user
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Error fingerprints ────────────────────────────────────────────────────────
# Patterns that indicate a DAILY / MONTHLY quota is exhausted — these will NOT
# recover by waiting a few seconds, so retrying is wasteful and expensive.
_DAILY_QUOTA_PATTERNS = (
    "daily limit",
    "daily quota",
    "quota exceeded",
    "you exceeded your current quota",
    "billing",
    "insufficient_quota",
    "exceeded your current quota",
    "resource_exhausted",                  # gRPC status from Gemini
    "rateLimitExceeded",
    "userRateLimitExceeded",
    "quota_exceeded",
    "free tier",
    "plan limit",
    "out of credits",
)

# Patterns that indicate a SHORT-TERM rate limit — these may recover; tenacity
# handles these via with_backoff(), but if all retries are exhausted we still
# want a friendly message.
_RATE_LIMIT_PATTERNS = (
    "429",
    "too many requests",
    "rate limit",
    "rate_limit",
    "resource exhausted",
    "try again",
    "retry after",
    "temporarily unavailable",
    "503",
    "service unavailable",
)


def classify_api_error(exc: BaseException) -> str | None:
    """Classify an exception as 'quota', 'rate_limit', or None (not API-related).

    Returns:
        'quota'      — daily/monthly limit exhausted; will not recover this session.
        'rate_limit' — short-term throttle; may recover after a wait.
        None          — not a recognised API quota/rate error.
    """
    msg = str(exc).lower()
    if any(p.lower() in msg for p in _DAILY_QUOTA_PATTERNS):
        return "quota"
    if any(p.lower() in msg for p in _RATE_LIMIT_PATTERNS):
        return "rate_limit"
    return None


def quota_safe_invoke(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Call fn(*args, **kwargs), catching quota/rate errors gracefully.

    On quota exhaustion: logs a WARNING with full error detail, returns a
    sentinel dict so the caller can surface a friendly CLI message.

    On rate-limit exhaustion (all tenacity retries spent): logs an ERROR,
    returns a sentinel dict.

    On any other exception: re-raises so existing error handling is unchanged.

    Returns:
        The function's return value on success.
        {"quota_error": True, "kind": "quota"|"rate_limit", "message": str}
        on quota/rate exhaustion.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        kind = classify_api_error(exc)
        if kind == "quota":
            logger.warning(
                "LLM daily quota exhausted",
                extra={"error": str(exc), "kind": kind},
            )
            from src.observability import metrics
            metrics.increment("llm_quota_exhausted")
            return {
                "quota_error": True,
                "kind": "quota",
                "message": (
                    "The AI service has reached its daily usage limit.\n"
                    "Please try again tomorrow, or ask your administrator to "
                    "check the API quota in Google AI Studio / GCP Console."
                ),
            }
        if kind == "rate_limit":
            logger.error(
                "LLM rate limit — all retries exhausted",
                extra={"error": str(exc), "kind": kind},
            )
            from src.observability import metrics
            metrics.increment("llm_rate_limit_exhausted")
            return {
                "quota_error": True,
                "kind": "rate_limit",
                "message": (
                    "The AI service is temporarily rate-limited and did not "
                    "recover after several retries.\n"
                    "Please wait a minute and try your question again."
                ),
            }
        raise
