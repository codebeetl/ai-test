"""Unit tests for SQL self-correction loop."""

import pytest
from src.resilience.sql_self_correct import with_sql_self_correction, MAX_SQL_RETRIES
import pandas as pd


def test_success_on_first_attempt():
    df = pd.DataFrame({"total": [100]})
    result = with_sql_self_correction(lambda sql: df, "SELECT 1")
    assert result["rows"] == [{"total": 100}]
    assert result["columns"] == ["total"]
    assert "error" not in result


def test_returns_error_dict_after_max_retries():
    def always_fail(sql):
        raise Exception("syntax error near FORM")

    result = with_sql_self_correction(always_fail, "SELECT * FORM orders")
    assert result["rows"] == []
    assert result["columns"] == []
    assert "error" in result


def test_warning_added_on_retry_success():
    call_count = {"n": 0}

    def fail_then_succeed(sql):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("first attempt fail")
        return pd.DataFrame({"count": [5]})

    # Patch rewriter to return a trivially different SQL
    import src.resilience.sql_self_correct as module
    original = module._rewrite_sql_with_llm
    module._rewrite_sql_with_llm = lambda sql, err: "SELECT COUNT(*) AS count FROM orders"

    try:
        result = with_sql_self_correction(fail_then_succeed, "BAD SQL")
        assert "warning" in result
        assert result["rows"] == [{"count": 5}]
    finally:
        module._rewrite_sql_with_llm = original
