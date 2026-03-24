"""Offline QA smoke tests — verify BigQuery queries return sane results (Req 6).

Run with:
    pytest tests/test_qa_evals.py

These tests hit the real BigQuery public dataset (read-only, free tier).
Ensure GCP credentials are configured before running.
"""

import pytest
from src.tools.query_tool import run_bigquery_query

GOLDEN_QUERIES = [
    {
        "description": "Total order count is a positive integer",
        "sql": (
            "SELECT COUNT(*) AS total "
            "FROM `bigquery-public-data.thelook_ecommerce.orders`"
        ),
        "expect_columns": ["total"],
        "expect_min_rows": 1,
        "expect_min_value": {"total": 1},
    },
    {
        "description": "Products table has id and name columns",
        "sql": (
            "SELECT id, name "
            "FROM `bigquery-public-data.thelook_ecommerce.products` "
            "LIMIT 5"
        ),
        "expect_columns": ["id", "name"],
        "expect_min_rows": 1,
        "expect_min_value": {},
    },
    {
        "description": "Monthly revenue query returns numeric revenue column",
        "sql": (
            "SELECT FORMAT_DATE('%Y-%m', DATE(o.created_at)) AS month, "
            "SUM(oi.sale_price) AS revenue "
            "FROM `bigquery-public-data.thelook_ecommerce.orders` o "
            "JOIN `bigquery-public-data.thelook_ecommerce.order_items` oi "
            "  ON o.order_id = oi.order_id "
            "GROUP BY month ORDER BY month DESC LIMIT 12"
        ),
        "expect_columns": ["month", "revenue"],
        "expect_min_rows": 1,
        "expect_min_value": {},
    },
]


@pytest.mark.parametrize("case", GOLDEN_QUERIES, ids=[c["description"] for c in GOLDEN_QUERIES])
def test_golden_query(case):
    result = run_bigquery_query.invoke({"sql": case["sql"]})
    assert not result.get("error"), f"Query failed: {result.get('error')}"
    assert result["columns"] == case["expect_columns"], (
        f"Expected columns {case['expect_columns']}, got {result['columns']}"
    )
    assert len(result["rows"]) >= case["expect_min_rows"], (
        f"Expected >= {case['expect_min_rows']} rows, got {len(result['rows'])}"
    )
    for col, min_val in case.get("expect_min_value", {}).items():
        actual = result["rows"][0].get(col)
        assert actual is not None and actual >= min_val, (
            f"Column '{col}' value {actual} is below minimum {min_val}"
        )


def test_pii_columns_not_returned():
    """PII columns should never appear in query results (Req 2)."""
    result = run_bigquery_query.invoke({
        "sql": (
            "SELECT id, email, first_name "
            "FROM `bigquery-public-data.thelook_ecommerce.users` LIMIT 5"
        )
    })
    pii_cols = {"email", "phone", "phone_number", "mobile"}
    returned_cols = {c.lower() for c in result.get("columns", [])}
    leaked = pii_cols & returned_cols
    # mask_dataframe_pii drops these — after agent processing none should appear
    assert not leaked, f"PII columns leaked into output: {leaked}"
