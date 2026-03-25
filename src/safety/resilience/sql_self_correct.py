"""SQL self-correction loop — retries failed queries via LLM rewriting.

On a BigQuery error, the original SQL and error message are fed back to
the LLM with a correction prompt. Capped at MAX_SQL_RETRIES to prevent
runaway token spend.
"""

import re as _re
import logging
from typing import Callable

from src.config.settings import settings   # ← single import, singleton

logger = logging.getLogger(__name__)

MAX_SQL_RETRIES = settings.agent.max_sql_retries   # ← was hardcoded 2, now from config
_DATASET = settings.bigquery.dataset               # ← was hardcoded string, now from config
_TABLES = settings.bigquery.tables                 # ← was hardcoded list, now from config


def _extract_text(content) -> str:
    """Safely extract string from LLM response content (list or string).

    Gemini models may return content as a list of content blocks rather
    than a plain string. This normalises both cases.
    """
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ).strip()
    return str(content).strip()


def _sanitise_table_refs(sql: str) -> str:
    """Replace unqualified table references with fully-qualified BigQuery names.

    Uses negative lookbehind (?<![`\\w]) and lookahead (?!\\w) to enforce
    true word boundaries, preventing matches inside column aliases such as
    'total_orders' or 'num_order_items'.

    Handles patterns like:
      orders                    → `bigquery-public-data.thelook_ecommerce.orders`
      your_dataset.orders       → `bigquery-public-data.thelook_ecommerce.orders`
      thelook_ecommerce.orders  → `bigquery-public-data.thelook_ecommerce.orders`
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
    """Wrap a BigQuery execution function with sanitisation and LLM-driven self-correction.

    Flow:
      1. Sanitise table references before first attempt.
      2. Execute SQL.
      3. On failure, ask the LLM to rewrite and sanitise again.
      4. Repeat up to MAX_SQL_RETRIES times.
      5. Return a graceful error dict if all retries are exhausted.

    Args:
        execute_fn: Callable accepting a SQL string, returning a DataFrame.
        sql: Initial SQL query string.

    Returns:
        dict with rows, columns, and optionally warning or error.
    """
    attempt = 0
    current_sql = _sanitise_table_refs(sql)

    while attempt <= MAX_SQL_RETRIES:
        try:
            df = execute_fn(current_sql)
            result = {"rows": df.to_dict(orient="records"), "columns": list(df.columns)}
            if attempt > 0:
                result["debug"] = f"Query succeeded after {attempt} self-correction(s)"
            return result
        except Exception as e:
            attempt += 1
            logger.debug(
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
    """Ask the LLM to fix a broken SQL query given the BigQuery error.

    Args:
        sql: The SQL string that failed.
        error_msg: Error returned by BigQuery.

    Returns:
        A corrected SQL string.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from src.resilience.retry import with_backoff as _backoff

    llm = ChatGoogleGenerativeAI(
        model=settings.llm.model.name,              # ← from config.yaml
        temperature=settings.llm.parameters.temperature,
    )

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
         "  - The orders table has no 'num_item' column; "
         "count items via order_items using COUNT(oi.id) or SUM(oi.sale_price)\n"
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
