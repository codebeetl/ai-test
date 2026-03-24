"""PII masking layer — applied to ALL agent outputs before CLI display.

Strategy: two-pass.
  Pass 1 — Regex: Fast, deterministic removal of email and phone patterns.
  Pass 2 — Column-level: Drops known PII columns from DataFrames before
            serialisation. Cheap and deterministic — no LLM cost.
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")

PII_COLUMNS = {"email", "phone", "phone_number", "mobile", "address"}


def mask_pii(text: str) -> str:
    """Replace PII tokens in a string with safe placeholders.

    Args:
        text: Raw agent output string, possibly containing PII.

    Returns:
        Sanitised string with PII replaced by [EMAIL REDACTED] / [PHONE REDACTED].
    """
    masked = _EMAIL_RE.sub("[EMAIL REDACTED]", text)
    masked = _PHONE_RE.sub("[PHONE REDACTED]", masked)
    if masked != text:
        logger.warning("PII detected and masked in output string")
    return masked


def mask_dataframe_pii(df: Any) -> Any:
    """Drop known PII columns from a pandas DataFrame before serialisation.

    Args:
        df: pandas DataFrame from a BigQuery result.

    Returns:
        DataFrame with PII columns removed.
    """
    cols_to_drop = [c for c in df.columns if c.lower() in PII_COLUMNS]
    if cols_to_drop:
        logger.warning("Dropping PII columns from DataFrame", extra={"cols": cols_to_drop})
        df = df.drop(columns=cols_to_drop)
    return df
