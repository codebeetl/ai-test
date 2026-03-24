"""SQL self-correction loop — retries failed queries using LLM rewriting.

When BigQuery returns a syntax or runtime error, this module feeds the
original SQL and the error message back to the LLM with a targeted
correction prompt. The corrected query is then re-executed.

Bounds and cost control:
  The loop runs at most MAX_SQL_RETRIES times. After that, it returns a
  structured error dict rather than raising — keeping the CLI stable and
  preventing runaway token spend on difficult queries.

  All retry attempts are logged at WARNING level with the attempt number
  and error message for observability / cost auditing.

LLM coupling:
  The actual LLM rewrite call is injected at graph construction time via
  set_rewriter(). This keeps the module independently testable without
  requiring a live LLM connection.
"""

from __future__ import annotations

import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

MAX_SQL_RETRIES = 2

# Injected at startup by graph.py — see set_rewriter() below.
_llm_rewriter: Callable[[str, str], str] | None = None


def set_rewriter(rewriter_fn: Callable[[str, str], str]) -> None:
    """Inject the LLM SQL-rewriter function at graph construction time.

    Called once during graph.py build_graph() to avoid circular imports
    and to keep this module testable with a mock rewriter.

    Args:
        rewriter_fn: Callable(sql: str, error: str) -> corrected_sql: str
    """
    global _llm_rewriter
    _llm_rewriter = rewriter_fn


def with_sql_self_correction(
    execute_fn: Callable[[str], Any],
    sql: str,
) -> dict:
    """Wrap a BigQuery execution callable with an LLM-driven self-correction loop.

    On the first failure, calls the injected LLM rewriter to produce a
    corrected SQL string, then retries. Returns a structured result dict
    in all cases — the caller never sees an unhandled exception.

    Args:
        execute_fn: Callable that accepts a SQL string and returns a DataFrame.
            Typically BigQueryRunner.execute_query.
        sql: The initial SQL query to attempt.

    Returns:
        On success: ``{'rows': list[dict], 'columns': list[str]}``
        On partial success (with retries): same + ``'warning': str``
        On total failure: ``{'error': str, 'rows': [], 'columns': []}``
    """
    current_sql = sql

    for attempt in range(MAX_SQL_RETRIES + 1):
        try:
            df = execute_fn(current_sql)
            result: dict = {
                "rows": df.to_dict(orient="records"),
                "columns": list(df.columns),
            }
            if attempt > 0:
                result["warning"] = (
                    f"Query succeeded after {attempt} LLM correction(s). "
                    "The original SQL had errors."
                )
                logger.info(
                    "SQL self-correction succeeded",
                    extra={"attempts_needed": attempt},
                )
            return result

        except Exception as exc:
            logger.warning(
                "SQL execution failed — attempting LLM rewrite",
                extra={"attempt": attempt + 1, "error": str(exc)},
            )

            if attempt >= MAX_SQL_RETRIES:
                logger.error(
                    "SQL self-correction exhausted all retries",
                    extra={"max_retries": MAX_SQL_RETRIES, "final_error": str(exc)},
                )
                return {
                    "error": (
                        f"Query failed after {MAX_SQL_RETRIES} correction attempts. "
                        f"Last error: {exc}"
                    ),
                    "rows": [],
                    "columns": [],
                }

            if _llm_rewriter is None:
                return {
                    "error": "LLM rewriter not configured and query failed.",
                    "rows": [],
                    "columns": [],
                }

            try:
                current_sql = _llm_rewriter(current_sql, str(exc))
                logger.info(
                    "LLM produced corrected SQL",
                    extra={"attempt": attempt + 1},
                )
            except Exception as rewrite_exc:
                return {
                    "error": f"SQL rewriter failed: {rewrite_exc}",
                    "rows": [],
                    "columns": [],
                }

    # Should be unreachable, but satisfies type checker
    return {"error": "Unexpected loop exit", "rows": [], "columns": []}
