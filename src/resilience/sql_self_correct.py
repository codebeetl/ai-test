"""SQL self-correction loop — retries failed queries via LLM rewriting.

All tunable parameters (max retries, PII columns, correction model) are read
from config.yaml via load_settings() so they can be changed without touching code.
"""

import re as _re
import logging
from typing import Callable

logger = logging.getLogger(__name__)

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
    """Replace unqualified table references with fully-qualified BigQuery names."""
    for table in _TABLES:
        pattern = _re.compile(
            r"(?<![`\w])`?(?:[\w\-]+\.)*" + _re.escape(table) + r"`?(?!\w)",
            _re.IGNORECASE,
        )
        replacement = f"`{_DATASET}.{table}`"
        sql = pattern.sub(replacement, sql)
    return sql


def _flag_pii_sql(sql: str, pii_cols: set[str]) -> str:
    """Log a warning if PII column names appear in the SQL (defence-in-depth)."""
    lower = sql.lower()
    detected = [col for col in pii_cols if _re.search(rf"\b{col}\b", lower)]
    if detected:
        logger.warning(
            "PII columns detected in SQL before execution",
            extra={"cols": detected},
        )
    return sql


def with_sql_self_correction(execute_fn: Callable, sql: str) -> dict:
    """Wrap a BigQuery execution function with sanitisation and LLM-driven self-correction.

    Max retries and PII columns are read from config.yaml on each call.
    """
    from src.config.settings import load_settings
    settings = load_settings()
    max_retries = settings.resilience.sql_max_retries
    pii_cols = set(settings.safety.pii_columns)

    attempt = 0
    current_sql = _flag_pii_sql(_sanitise_table_refs(sql), pii_cols)

    while attempt <= max_retries:
        try:
            df = execute_fn(current_sql)
            result = {"rows": df.to_dict(orient="records"), "columns": list(df.columns)}
            if attempt > 0:
                result["warning"] = f"Query succeeded after {attempt} self-correction(s)"
            return result
        except Exception as e:
            attempt += 1
            logger.debug(
                "SQL attempt failed",
                extra={"attempt": attempt, "error": str(e)[:200]},
            )
            if attempt > max_retries:
                logger.debug("Max SQL retries exceeded, returning error dict")
                return {"error": str(e), "rows": [], "columns": []}
            try:
                rewritten = _rewrite_sql_with_llm(current_sql, str(e))
                current_sql = _flag_pii_sql(_sanitise_table_refs(rewritten), pii_cols)
            except Exception as rewrite_err:
                logger.error(f"SQL rewriter failed: {rewrite_err}")
                return {"error": str(e), "rows": [], "columns": []}

    return {"error": "Max retries exceeded", "rows": [], "columns": []}


def _rewrite_sql_with_llm(sql: str, error_msg: str) -> str:
    """Ask the LLM to fix a broken SQL query — uses correction_model from config.yaml."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from src.resilience.retry import with_backoff as _backoff
    from src.config.settings import load_settings

    settings = load_settings()
    llm = ChatGoogleGenerativeAI(
        model=settings.llm.correction_model,
        temperature=0,
    )

    @_backoff(
        max_attempts=settings.resilience.llm_max_attempts,
        min_wait=settings.resilience.llm_min_wait_s,
        max_wait=settings.resilience.llm_max_wait_s,
    )
    def _call(msgs):
        return llm.invoke(msgs)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Fix the following BigQuery SQL based on the error message.\n"
         f"Always use fully-qualified table names: `{_DATASET}.table_name`\n"
         "\nExact column names available:\n"
         "  orders:      order_id, user_id, status, gender, created_at, num_of_item\n"
         "  order_items: id, order_id, user_id, product_id, status, created_at, sale_price\n"
         "  products:    id, cost, category, name, brand, retail_price, department, sku\n"
         "  users:       id, age, gender, state, city, country, traffic_source, created_at\n"
         "\nCommon fixes:\n"
         "  - Ensure every opening parenthesis has a matching closing parenthesis\n"
         "  - Use DATE(created_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL N DAY) for dates\n"
         "  - Cast TIMESTAMP to DATE with DATE() before using DATE_SUB\n"
         "  - Revenue = SUM(oi.sale_price) joining order_items to orders on order_id\n"
         "  - For country queries join users on user_id, use u.country\n"
         "  - Never select PII columns (email, phone, phone_number, mobile, address,\n"
         "    first_name, last_name)\n"
         "  - Every subquery must have an alias\n"
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
