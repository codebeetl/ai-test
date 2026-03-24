"""PII masking layer — applied to ALL agent outputs before CLI display.

Strategy: two-pass approach.

  Pass 1 — Regex (always runs):
    Fast, deterministic removal of email addresses and phone number patterns.
    Covers international formats (+44, +1, etc.) and common email structures.

  Pass 2 — Column-level drop (DataFrame inputs):
    Removes columns definitively identified as PII in the thelook_ecommerce
    schema (users.email, users.phone). Operates on column *names* rather than
    content so it is cheap and deterministic regardless of data volume.

  The LLM-based indirect-PII scan (e.g. catching 'customer John at john@...')
  is architecturally planned but intentionally omitted from this prototype to
  keep the scope focused. It would be inserted between Pass 1 and return as
  Pass 3, gated behind a heuristic to avoid inflating token costs.

Safety guarantee:
    mask_pii() and mask_dataframe_pii() are the *only* exit points for data
    leaving the agent graph. graph.py routes ALL tool results through
    mask_and_format() which calls both functions before setting final_output.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9.\-]+"
)

# Matches international phone numbers (+44 161 555 0192, (800) 555-1234, etc.)
_PHONE_RE = re.compile(
    r"(\+?\d[\d\s\-().]{7,}\d)"
)

# Column names that are definitively PII in the thelook_ecommerce schema
_PII_COLUMNS: frozenset[str] = frozenset(
    {"email", "phone", "phone_number", "mobile", "street_address", "postal_code"}
)


def mask_pii(text: str) -> str:
    """Replace PII tokens in a string with safe placeholders.

    Runs the regex pass unconditionally. Logs a WARNING when PII is detected
    so the observability layer can alert on unexpected data leakage patterns.

    Args:
        text: Raw agent output string, possibly containing PII values
            sourced from BigQuery results.

    Returns:
        Sanitised string with all detected PII replaced by
        ``[EMAIL REDACTED]`` or ``[PHONE REDACTED]``.
    """
    masked = _EMAIL_RE.sub("[EMAIL REDACTED]", text)
    masked = _PHONE_RE.sub("[PHONE REDACTED]", masked)

    if masked != text:
        logger.warning(
            "PII detected and masked in agent output",
            extra={"pii_types_found": _detect_pii_types(text)},
        )
    return masked


def mask_dataframe_pii(df: "pd.DataFrame") -> "pd.DataFrame":
    """Drop known PII columns from a pandas DataFrame before serialisation.

    This is the primary defence for structured query results. Column-name
    matching is intentionally strict (lowercase comparison) to avoid accidental
    retention under aliased names.

    Args:
        df: pandas DataFrame returned directly from BigQueryRunner.execute_query.

    Returns:
        A new DataFrame with all PII-identified columns removed. The original
        DataFrame is not mutated.
    """
    cols_to_drop = [c for c in df.columns if c.lower() in _PII_COLUMNS]
    if cols_to_drop:
        logger.warning(
            "Dropping PII columns from DataFrame before output",
            extra={"dropped_columns": cols_to_drop},
        )
        return df.drop(columns=cols_to_drop)
    return df


def _detect_pii_types(text: str) -> list[str]:
    """Helper: identify which PII types were present for structured logging."""
    found = []
    if _EMAIL_RE.search(text):
        found.append("email")
    if _PHONE_RE.search(text):
        found.append("phone")
    return found
