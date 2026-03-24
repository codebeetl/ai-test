"""SQL self-correction loop — retries failed queries via LLM rewriting.

On a BigQuery error, the original SQL and the error message are fed back
to the LLM with a correction prompt. The loop runs at most MAX_SQL_RETRIES
times to prevent runaway token spend.
"""

import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

MAX_SQL_RETRIES = 2


def with_sql_self_correction(execute_fn: Callable, sql: str) -> dict:
    """Wrap a BigQuery execution function with LLM-driven self-correction.

    Args:
        execute_fn: Callable accepting a SQL string, returning a DataFrame.
        sql: Initial SQL query string.

    Returns:
        dict with 'rows', 'columns', and optionally 'warning' or 'error'.
    """
    attempt = 0
    current_sql = sql

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
                return {"error": str(e), "rows": [], "columns": []}
            current_sql = _rewrite_sql_with_llm(current_sql, str(e))

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

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Fix the following BigQuery SQL based on the error message.\n"
         "Return ONLY the corrected SQL. No explanation, no markdown."),
        ("user", "SQL:\n{sql}\n\nError:\n{error}"),
    ])
    msg = prompt.format_messages(sql=sql, error=error_msg)
    resp = llm.invoke(msg)
    return resp.content.strip().strip("```sql").strip("```").strip()
