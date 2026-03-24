"""Unit tests for PII masking layer."""

import pandas as pd
import pytest
from src.safety.pii_masker import mask_pii, mask_dataframe_pii


def test_email_is_masked():
    result = mask_pii("Contact jane@example.com for details")
    assert "jane@example.com" not in result
    assert "[EMAIL REDACTED]" in result


def test_phone_is_masked():
    result = mask_pii("Call +44 161 555 0192 now")
    assert "+44 161 555 0192" not in result
    assert "[PHONE REDACTED]" in result


def test_clean_text_unchanged():
    text = "Total revenue was £12,500 this quarter"
    assert mask_pii(text) == text


def test_dataframe_drops_email_column():
    df = pd.DataFrame({"name": ["Alice"], "email": ["alice@test.com"], "revenue": [100]})
    result = mask_dataframe_pii(df)
    assert "email" not in result.columns
    assert "name" in result.columns
    assert "revenue" in result.columns


def test_dataframe_drops_phone_column():
    df = pd.DataFrame({"customer_id": [1], "phone": ["07700900000"], "total": [50]})
    result = mask_dataframe_pii(df)
    assert "phone" not in result.columns


def test_dataframe_no_pii_columns_unchanged():
    df = pd.DataFrame({"product": ["Widget"], "sales": [42]})
    result = mask_dataframe_pii(df)
    assert list(result.columns) == ["product", "sales"]
