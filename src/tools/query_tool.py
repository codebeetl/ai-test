"""BigQuery execution tool — the agent's primary data access point."""

import logging
from langchain_core.tools import tool
from src.db.bq_client import BigQueryRunner
from src.resilience.sql_self_correct import with_sql_self_correction
from src.resilience.retry import with_backoff
from src.config.settings import load_settings

logger = logging.getLogger(__name__)
_bq = BigQueryRunner()


def _execute_with_retry(sql: str):
    """Execute BigQuery with back-off parameters read from config.yaml."""
    settings = load_settings()
    r = settings.resilience

    @with_backoff(
        max_attempts=r.bq_max_attempts,
        min_wait=r.bq_min_wait_s,
        max_wait=r.bq_max_wait_s,
    )
    def _inner(s: str):
        return _bq.execute_query(s)

    return _inner(sql)


@tool
def run_bigquery_query(sql: str) -> dict:
    """Execute a read-only BigQuery SQL query against thelook_ecommerce.

    Wrapped in a self-correction loop: if BigQuery returns an error,
    the LLM rewrites the SQL up to sql_max_retries times (config.yaml)
    before surfacing a graceful failure dict.

    Args:
        sql: A valid BigQuery SQL string targeting the thelook_ecommerce dataset.

    Returns:
        dict with keys 'rows' (list[dict]) and 'columns' (list[str]).
        Returns {'error': str, 'rows': [], 'columns': []} on unrecoverable failure.
    """
    logger.info("run_bigquery_query invoked", extra={"sql_preview": sql[:120]})
    return with_sql_self_correction(_execute_with_retry, sql)
