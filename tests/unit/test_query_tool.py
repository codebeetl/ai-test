"""Unit tests for BigQuery query tool with mocked runner."""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


def test_run_bigquery_query_returns_dict():
    mock_df = pd.DataFrame({"orders": [42]})

    with patch("src.tools.query_tool._bq") as mock_bq:
        mock_bq.execute_query.return_value = mock_df
        from src.tools.query_tool import run_bigquery_query
        result = run_bigquery_query.invoke({"sql": "SELECT COUNT(*) AS orders FROM orders"})
        assert result["rows"] == [{"orders": 42}]
        assert result["columns"] == ["orders"]


def test_run_bigquery_query_graceful_on_error():
    with patch("src.tools.query_tool._bq") as mock_bq:
        mock_bq.execute_query.side_effect = Exception("Table not found")
        from src.tools.query_tool import run_bigquery_query
        result = run_bigquery_query.invoke({"sql": "SELECT * FROM nonexistent_table"})
        assert "error" in result
        assert result["rows"] == []
