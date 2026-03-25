"""PII masking layer — applied to ALL agent outputs before CLI display.

Strategy: two-pass.
  Pass 1 — Regex: Fast, deterministic removal of email and phone patterns.
  Pass 2 — Column-level: Drops PII columns from DataFrames before serialisation.

PII column names are configured in config.yaml under safety.pii_columns so
the list can be extended without touching code.
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
# Require at least one formatting character (space, dash, dot, parens) to avoid
# false positives on revenue figures, product IDs, and other numeric data.
_PHONE_RE = re.compile(
    r"(\+?\d[\d]{1,3}[\s\-.()]+[\d\s\-.()]{5,}\d)"
)


def _get_pii_columns() -> set[str]:
    """Read PII column names from config.yaml — called fresh each time so
    changes to config.yaml take effect without restarting the process."""
    from src.config.settings import load_settings
    return {c.lower() for c in load_settings().safety.pii_columns}


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
        logger.debug("PII detected and masked in output string")
    return masked


def mask_dataframe_pii(df: Any) -> Any:
    """Drop PII columns from a pandas DataFrame before serialisation.

    The column list is read from config.yaml (safety.pii_columns) on each
    call so it can be extended without restarting the process.

    Args:
        df: pandas DataFrame from a BigQuery result.

    Returns:
        DataFrame with PII columns removed.
    """
    pii_columns = _get_pii_columns()
    cols_to_drop = [c for c in df.columns if c.lower() in pii_columns]
    if cols_to_drop:
        logger.warning(
            "Dropping PII columns from DataFrame",
            extra={"cols": cols_to_drop},
        )
        df = df.drop(columns=cols_to_drop)
    return df
