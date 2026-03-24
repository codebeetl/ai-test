"""SQL self-correction loop — retries failed queries via LLM rewriting."""

import re as _re
import logging
from typing import Callable

logger = logging.getLogger(__name__)

MAX_SQL_RETRIES = 2
_DATASET = "bigquery-public-data.thelook_ecommerce"
_TABLES = ["order_items", "orders", "products", "users"]  # order_items before orders


def _extract_text(content) -> str:
    """Safely extract string from LLM response content (list or string)."""
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ).strip()
    return str(content).strip()


def _sanitise_table_refs(sql: str) -> str:
    """Replace unqualified table references with fully-qualified BigQuery names.

    Uses negative lookbehind and lookahead to avoid matching table names that
    appear inside column aliases (e.g. 'total_orders', 'num_order_items').
    """
    for table in _TABLES:
        pattern = _re.compile(
            r"(?<![`\w])`?(?:[\w\-]+\.)*" + _re.escape(table) + r"`?(?!\w)",
            _re.IGNORECASE,
        )
        replacement = f"`{_DATASET}.{table}`"
        sql = pattern.sub(replacement, sql)
    return sql


def with_sql_self_correction(execute_fn: Callable, sql: str) -> dict:
    """Wrap a BigQuery execution function with sanitisation and LLM-driven self-correction."""
    attempt = 0
    current_sql = _sanitise_table_refs(sql)

    while attempt <= MAX_SQL_RETRIES:
        try:
            df = execute_fn(current_sql)
            result = {"rows": df.to_dict(orient="records"), "columns": list(df.columns)}
            if attempt > 0:
                result["warning"] = f"Query succeeded after {attempt} self-correction(s)"
            return result
        except Exception as e:
            attempt += 1
            logger.warning(
                "SQL attempt failed",
                extra={"attempt": attempt, "error": str(e)[:200]},
            )
            if attempt > MAX_SQL_RETRIES:
                logger.error("Max SQL retries exceeded, returning error dict")
                return {"error": str(e), "rows": [], "columns": []}
            try:
                rewritten = _rewrite_sql_with_llm(current_sql, str(e))
                current_sql = _sanitise_table_refs(rewritten)
            except Exception as rewrite_err:
                logger.error(f"SQL rewriter failed: {rewrite_err}")
                return {"error": str(e), "rows": [], "columns": []}

    return {"error": "Max retries exceeded", "rows": [], "columns": []}


def _rewrite_sql_with_llm(sql: str, error_msg: str) -> str:
    """Ask the LLM to fix a broken SQL query given the BigQuery error."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from src.resilience.retry import with_backoff as _backoff

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    @_backoff(max_attempts=3, min_wait=5, max_wait=30)
    def _call(msgs):
        return llm.invoke(msgs)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Fix the following BigQuery SQL based on the error message.\n"
         f"Always use fully-qualified table names: `{_DATASET}.table_name`\n"
         "Common fixes:\n"
         "  - Use DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR) for date ranges\n"
         "  - Cast TIMESTAMP to DATE with DATE() before using DATE_SUB\n"
         "  - Use TIMESTAMP_SUB only with HOUR, MINUTE, SECOND, MICROSECOND\n"
         "Return ONLY the corrected SQL. No explanation, no markdown."),
        ("user", "SQL:\n{sql}\n\nError:\n{error}"),
    ])
    msg = prompt.format_messages(sql=sql, error=error_msg)
    resp = _call(msg)
    sql_out = _extract_text(resp.content)
    if sql_out.startswith("```sql"):
        sql_out = sql_out[6:]
    elif sql_out.startswith("```"):
        sql_out = sql_out[3:]
    if sql_out.endswith("```"):
        sql_out = sql_out[:-3]
    return sql_out.strip()
