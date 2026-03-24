"""Schema introspection tool — lets the agent answer structural questions."""

import logging
from langchain_core.tools import tool
from src.db.bq_client import BigQueryRunner

logger = logging.getLogger(__name__)
_bq = BigQueryRunner()

SUPPORTED_TABLES = ["orders", "order_items", "products", "users"]


@tool
def get_table_schema(table_name: str) -> list[dict]:
    """Retrieve the schema for a thelook_ecommerce table.

    Use this when the user asks about the database structure, available
    columns, or data types.

    Args:
        table_name: One of: orders, order_items, products, users.

    Returns:
        List of dicts with keys 'name', 'type', 'mode', 'description'.
    """
    if table_name not in SUPPORTED_TABLES:
        return [{"error": f"Unknown table. Supported: {SUPPORTED_TABLES}"}]
    logger.info("Schema requested", extra={"table": table_name})
    return _bq.get_table_schema(table_name)
