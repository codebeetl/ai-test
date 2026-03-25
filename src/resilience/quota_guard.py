"""Quota and rate-limit guard for LLM and external API calls.

Distinguishes between:
  - DAILY quota exhaustion  — will NOT recover this session; clear user message.
  - TPM/RPM soft rate limit — recovers after a short wait; tenacity handles
                              retries, but if all retries are spent we show
                              the exact wait time from the error response.

Usage:
    from src.resilience.quota_guard import quota_safe_invoke

    response = quota_safe_invoke(_invoke_llm, prompt_messages)
    if isinstance(response, dict) and response.get("quota_error"):
        # surface response["message"] to the user
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Daily / permanent quota fingerprints ─────────────────────────────────────
# These patterns are SPECIFIC to per-day or per-month exhaustion.
# They must NOT match short-term TPM/RPM throttles.
_DAILY_QUOTA_PATTERNS = (
    "per_day",
    "perday",
    "per day",
    "GenerateRequestsPerDayPerProjectPerModel",   # exact Gemini quotaId
    "generate_content_free_tier_requests",        # exact Gemini free-tier metric
    "free_tier_requests",
    "daily_request",
    "requests_per_day",
    "RequestsPerDay",
    "you exceeded your current quota",            # Gemini billing message
    "check your plan and billing",
    "insufficient_quota",
    "out of credits",
    "plan limit",
)

# ── Short-term / soft rate-limit fingerprints ─────────────────────────────────
# TPM = tokens per minute, RPM = requests per minute.
# Tenacity retries these automatically; this guard only fires when ALL retries
# are exhausted, at which point we extract the suggested wait time if present.
_SOFT_RATE_LIMIT_PATTERNS = (
    "per_minute",
    "perminute",
    "per minute",
    "GenerateRequestsPerMinute",
    "GenerateTokensPerMinute",
    "tokens_per_minute",
    "requests_per_minute",
    "too many requests",
    "rate limit",
    "rate_limit",
    "retry after",
    "retry_after",
    "temporarily unavailable",
    "503",
    "service unavailable",
)


def _extract_retry_delay(error_str: str) -> str | None:
    """Parse the suggested retry delay from a Gemini/Google API error string.

    Looks for patterns like:
      "retry in 9.26s"
      "retryDelay: 9s"
      "Retry-After: 30"
    Returns a human-readable string like "9 seconds" or None if not found.
    """
    patterns = [
        r"retry[\s_-]*(?:in|after|delay)[:\s]*([\d.]+)\s*s",   # retry in 9.26s / retryDelay: 9s
        r"please retry in ([\d.]+)s",                              # Gemini: "Please retry in 9s"
        r"retry-after[:\s]+(\d+)",                                # HTTP header style
        r"wait ([\d.]+) second",                                   # plain English
    ]
    lower = error_str.lower()
    for pattern in patterns:
        m = re.search(pattern, lower)
        if m:
            secs = float(m.group(1))
            # Round up to nearest second for display
            secs_int = int(secs) + (1 if secs % 1 > 0 else 0)
            return f"{secs_int} second{'s' if secs_int != 1 else ''}"
    return None


def classify_api_error(exc: BaseException) -> str | None:
    """Classify an exception as 'daily_quota', 'rate_limit', or None.

    Returns:
        'daily_quota'  — per-day/month limit exhausted; will not recover this session.
        'rate_limit'   — short-term TPM/RPM throttle; recovers after a brief wait.
        None            — not a recognised API quota/rate error; re-raise as normal.
    """
    msg = str(exc)
    msg_lower = msg.lower()

    # Daily quota check — use case-insensitive search against the original message
    # to catch CamelCase quota IDs like GenerateRequestsPerDayPerProjectPerModel
    for pattern in _DAILY_QUOTA_PATTERNS:
        if pattern.lower() in msg_lower:
            return "daily_quota"

    # Soft rate-limit check
    for pattern in _SOFT_RATE_LIMIT_PATTERNS:
        if pattern.lower() in msg_lower:
            return "rate_limit"

    # Catch bare 429 only if no daily pattern matched above
    if "429" in msg:
        return "rate_limit"

    return None


def _daily_quota_message(error_str: str = "") -> str:
    """Build a user-friendly daily quota message with the reset time."""
    from datetime import datetime, timezone, timedelta

    # Gemini free-tier resets at midnight US Pacific (UTC-7 PDT / UTC-8 PST)
    # Approximate: use UTC-7 as conservative estimate
    now_utc = datetime.now(timezone.utc)
    pacific_offset = timedelta(hours=-7)
    now_pacific = now_utc + pacific_offset
    reset_pacific = (now_pacific + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    reset_utc = reset_pacific - pacific_offset
    hours_until = (reset_utc - now_utc).seconds // 3600
    mins_until = ((reset_utc - now_utc).seconds % 3600) // 60

    if hours_until > 0:
        wait_str = f"approximately {hours_until}h {mins_until}m"
    else:
        wait_str = f"approximately {mins_until} minute(s)"

    reset_local = reset_utc.strftime("%H:%M UTC")

    return (
        f"⚠️  Daily API quota reached.\n"
        f"You have exhausted the free-tier daily request limit for the Gemini API.\n"
        f"Quota resets in {wait_str} (at {reset_local}).\n"
        f"Alternatively, switch to a paid API key to continue immediately.\n"
        f"Details: https://ai.google.dev/gemini-api/docs/rate-limits"
    )


def quota_safe_invoke(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Call fn(*args, **kwargs), catching quota/rate errors gracefully.

    - Daily quota exhausted  → friendly "try again tomorrow" message.
    - TPM/RPM all retries spent → friendly "wait N seconds" message with
      the exact retry delay extracted from the API response where available.
    - Any other exception    → re-raised unchanged.

    Returns:
        fn's return value on success.
        {"quota_error": True, "kind": str, "message": str} on quota/rate error.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        from src.observability import metrics

        kind = classify_api_error(exc)

        if kind == "daily_quota":
            logger.warning(
                "LLM daily quota exhausted",
                extra={"error": str(exc)[:400], "kind": kind},
            )
            metrics.increment("llm_daily_quota_exhausted")
            return {
                "quota_error": True,
                "kind": "daily_quota",
                "message": _daily_quota_message(str(exc)),
            }

        if kind == "rate_limit":
            retry_delay = _extract_retry_delay(str(exc))
            wait_hint = (
                f"Please wait {retry_delay} and try again."
                if retry_delay
                else "Please wait a moment and try again."
            )
            logger.warning(
                "LLM rate limit — all retries exhausted",
                extra={"error": str(exc)[:400], "kind": kind, "retry_delay": retry_delay},
            )
            metrics.increment("llm_rate_limit_exhausted")
            return {
                "quota_error": True,
                "kind": "rate_limit",
                "message": (
                    f"⏱️  Rate limit reached (requests or tokens per minute).\n"
                    f"{wait_hint}"
                ),
            }

        raise
