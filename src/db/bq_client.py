"""A lean BigQuery client for executing SQL queries and returning DataFrame results."""

import logging
from typing import Optional, List, Dict, Any
import pandas as pd
from google.cloud import bigquery

from src.resilience.retry import with_backoff

logger = logging.getLogger(__name__)


class BigQueryRunner:
    """Thin wrapper around the BigQuery Python client."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset_id: Optional[str] = "bigquery-public-data.thelook_ecommerce",
    ) -> None:
        logger.info("Initialising BigQuery client")
        try:
            self.client = bigquery.Client(project=project_id)
            self.dataset_id = dataset_id
            logger.info(f"BigQuery client ready for dataset: {self.dataset_id}")
        except Exception as e:
            logger.error(f"Failed to initialise BigQuery client: {e}")
            raise

    @with_backoff(max_attempts=4, min_wait=2, max_wait=30)
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """Execute a SQL query with exponential back-off on transient errors."""
        try:
            logger.info("Executing BigQuery query")
            query_job = self.client.query(sql_query)
            df = query_job.result().to_dataframe()
            logger.info(f"Query completed, returned {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"BigQuery execution failed: {e}")
            raise

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get schema information for a specific table."""
        try:
            table_ref = f"{self.dataset_id}.{table_name}"
            table = self.client.get_table(table_ref)
            return [
                {
                    "name": field.name,
                    "type": field.field_type,
                    "mode": field.mode,
                    "description": field.description or "",
                }
                for field in table.schema
            ]
        except Exception as e:
            logger.error(f"Failed to get schema for {table_name}: {e}")
            raise
