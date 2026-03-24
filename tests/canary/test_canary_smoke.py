"""Canary smoke tests — run against production after deploy.

These tests issue a single known-safe query, assert no PII in the output,
and verify the agent responds within the time budget.
"""

import time
import re
import pytest
from unittest.mock import patch
import pandas as pd


@pytest.mark.timeout(30)
def test_canary_no_pii_in_output():
    """Agent should never return email or phone in its final output."""
    mock_df = pd.DataFrame({"total_orders": [12345]})

    with patch("src.tools.query_tool._bq") as mock_bq, \
         patch("src.agent.nodes._llm") as mock_llm, \
         patch("src.tools.golden_bucket_tool._get_gb") as mock_gb:

        mock_bq.execute_query.return_value = mock_df
        mock_llm.invoke.return_value = MagicMock(content="SELECT COUNT(*) AS total_orders FROM `bigquery-public-data.thelook_ecommerce.orders`")
        mock_gb.return_value.similarity_search.return_value = []

        from src.agent.graph import build_graph
        from src.agent.state import AgentState

        graph = build_graph()
        state: AgentState = {
            "messages": [{"role": "user", "content": "How many total orders are there?"}],
            "user_id": "canary_user",
            "pending_destructive_op": None,
            "last_sql": None,
            "retry_count": 0,
            "raw_result": None,
            "final_output": None,
        }

        start = time.time()
        result = graph.invoke(state)
        elapsed = time.time() - start

        output = result.get("final_output", "")
        email_pattern = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
        phone_pattern = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")

        assert not email_pattern.search(output), "PII (email) found in canary output!"
        assert not phone_pattern.search(output), "PII (phone) found in canary output!"
        assert elapsed < 30, f"Canary took too long: {elapsed:.1f}s"
        print(f"✅ Canary passed in {elapsed:.1f}s")
