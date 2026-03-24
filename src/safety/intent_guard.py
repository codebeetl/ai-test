"""Intent guard — rejects non-analysis queries before they reach the LLM.

The agent is strictly scoped to retail data *analysis* questions. This module
provides a fast keyword/pattern pre-filter that short-circuits clearly
out-of-scope or adversarial inputs (e.g. 'ignore previous instructions',
'tell me a joke', prompt injection attempts).

This is intentionally a denylist approach for the prototype. Production would
use an LLM-based intent classifier with a structured output schema.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_BLOCKED_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore (previous|all|prior) instructions", re.I),
    re.compile(r"(jailbreak|DAN mode|act as)", re.I),
    re.compile(r"(tell me a joke|write a poem|compose a song)", re.I),
    re.compile(r"(who are you|what is your name|are you human)", re.I),
]


def is_allowed_intent(user_message: str) -> bool:
    """Return True if the message passes the intent guard.

    Checks the user message against a set of blocked patterns that indicate
    the query is not a retail data analysis question. Logs any blocked attempt
    at WARNING level for security audit trails.

    Args:
        user_message: Raw user input string from the CLI.

    Returns:
        True if the message appears to be a valid analysis query.
        False if it matches a blocked pattern.
    """
    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(user_message):
            logger.warning(
                "Intent guard blocked query",
                extra={"pattern": pattern.pattern, "input_preview": user_message[:80]},
            )
            return False
    return True
