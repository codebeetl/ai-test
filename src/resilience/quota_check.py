"""Startup quota check — probes the Gemini API before the main loop begins.

Sends a minimal test request and interprets the result:
  - Success       → quota is available, print confirmation and continue.
  - Daily quota   → print reset time and exit(1) so the user is not misled.
  - Rate limit    → warn but continue (quota available, just throttled now).
  - Other error   → warn but continue (non-quota issue, e.g. network).

Call check_quota_or_exit() once at the top of main() before build_graph().
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

_PROBE_PROMPT = "Reply with the single word: ready"


def check_quota_or_exit() -> None:
    """Probe the LLM API at startup. Exits with code 1 if daily quota is exhausted."""
    from src.config.settings import load_settings
    from src.resilience.quota_guard import classify_api_error, _daily_quota_message
    from src.observability.progress import show as _progress, clear as _progress_clear

    _progress("Checking API quota...")

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        settings = load_settings()
        llm = ChatGoogleGenerativeAI(model=settings.llm.model, temperature=0)
        llm.invoke(_PROBE_PROMPT)
        _progress_clear()
        print("  ✅  API quota OK — ready to accept queries.")

    except Exception as exc:
        _progress_clear()
        kind = classify_api_error(exc)

        if kind == "daily_quota":
            msg = _daily_quota_message(str(exc))
            logger.warning("Startup quota check failed — daily quota exhausted", extra={"error": str(exc)[:300]})
            print()
            print("=" * 60)
            for line in msg.splitlines():
                print(f"  {line}")
            print("=" * 60)
            print()
            sys.exit(1)

        elif kind == "rate_limit":
            logger.warning("Startup quota check: rate-limited but quota available", extra={"error": str(exc)[:200]})
            print("  ⚠️   API is rate-limited right now but quota is available.")
            print("      Queries may be slow until the rate limit clears.")

        else:
            logger.warning("Startup quota check: unexpected error", extra={"error": str(exc)[:200]})
            print(f"  ⚠️   Could not verify API quota: {exc}")
            print("      Continuing anyway — first query will confirm connectivity.")
