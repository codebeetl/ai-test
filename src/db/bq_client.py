"""A lean BigQuery client for executing SQL queries and returning DataFrame results."""

import logging
from typing import Optional, List, Dict, Any
import pandas as pd
from google.cloud import bigquery

logger = logging.getLogger(__name__)


class BigQueryRunner:
    """Thin wrapper around the BigQuery Python client.

    Provides execute_query and get_table_schema, keeping all BigQuery
    interaction in one place so it is easy to mock in tests.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset_id: Optional[str] = "bigquery-public-data.thelook_ecommerce",
    ) -> None:
        """Initialise the BigQuery client.

        Args:
            project_id: GCP project ID. Uses application-default credentials if None.
            dataset_id: Default dataset for schema lookups.
        """
        logger.info("Initialising BigQuery client")
        try:
            self.client = bigquery.Client(project=project_id)
            self.dataset_id = dataset_id
            logger.info(f"BigQuery client ready for dataset: {self.dataset_id}")
        except Exception as e:
            logger.error(f"Failed to initialise BigQuery client: {e}")
            raise

    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame.

        Args:
            sql_query: The SQL query to execute.

        Returns:
            DataFrame containing the query results.

        Raises:
            Exception: If query execution fails.
        """
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
        """Get schema information for a specific table.

        Args:
            table_name: One of orders, order_items, products, users.

        Returns:
            List of dicts with column name, type, mode, description.
        """
        try:
            table_ref = f"{self.dataset_id}.{table_name}"
            table = self.client.get_table(table_ref)
            schema_info = []
            for field in table.schema:
                schema_info.append({
                    "name": field.name,
                    "type": field.field_type,
                    "mode": field.mode,
                    "description": field.description or "",
                })
            logger.info(f"Retrieved schema for table {table_name}")
            return schema_info
        except Exception as e:
            logger.error(f"Failed to get schema for {table_name}: {e}")
            raise
