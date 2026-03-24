"""BigQueryRunner — lean read-only BigQuery client for thelook_ecommerce.

This is an adaptation of the Opsfleet starter client, extended with:
  - Column-level PII masking applied at the DataFrame layer (before return)
  - Structured logging at INFO/ERROR level for observability
  - The resilient() retry decorator for transient GCP failures

All queries are READ-ONLY. This client has no mutation capability by design;
the dataset is a public BigQuery dataset and our GCP credentials are
configured with read-only IAM roles.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError

from src.resilience.retry import resilient
from src.safety.pii_masker import mask_dataframe_pii

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "bigquery-public-data.thelook_ecommerce"


class BigQueryRunner:
    """Read-only BigQuery client scoped to the thelook_ecommerce dataset.

    Wraps google-cloud-bigquery with retry logic, structured logging, and
    automatic PII column stripping on all returned DataFrames.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset_id: str = DEFAULT_DATASET,
    ) -> None:
        """Initialise the BigQuery client.

        Args:
            project_id: GCP project ID. If None, uses Application Default Credentials.
            dataset_id: Fully-qualified BigQuery dataset path.
        """
        logger.info("Initialising BigQuery client", extra={"dataset": dataset_id})
        try:
            self.client = bigquery.Client(project=project_id)
            self.dataset_id = dataset_id
            logger.info("BigQuery client ready", extra={"dataset": self.dataset_id})
        except Exception as exc:
            logger.error("Failed to initialise BigQuery client", extra={"error": str(exc)})
            raise

    @resilient(exception_types=(GoogleCloudError, TimeoutError))
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """Execute a SQL query and return a PII-masked DataFrame.

        PII columns (email, phone, etc.) are stripped at this layer as a
        defence-in-depth measure. The safety.pii_masker layer applies a
        second pass on the serialised string output before CLI display.

        Args:
            sql_query: A valid BigQuery Standard SQL string.

        Returns:
            pandas DataFrame with PII columns removed.

        Raises:
            GoogleCloudError: On unrecoverable BigQuery errors (after retries).
        """
        logger.info("Executing BigQuery query", extra={"sql_preview": sql_query[:120]})
        try:
            job = self.client.query(sql_query)
            df = job.result().to_dataframe()
            logger.info("Query completed", extra={"row_count": len(df)})
            return mask_dataframe_pii(df)
        except GoogleCloudError as exc:
            logger.error("BigQuery execution failed", extra={"error": str(exc)})
            raise

    def get_table_schema(self, table_name: str) -> list[dict[str, Any]]:
        """Retrieve schema metadata for a thelook_ecommerce table.

        Args:
            table_name: One of: orders, order_items, products, users.

        Returns:
            List of dicts with keys: name, type, mode, description.

        Raises:
            GoogleCloudError: If the table reference is invalid or access denied.
        """
        table_ref = f"{self.dataset_id}.{table_name}"
        try:
            table = self.client.get_table(table_ref)
            schema = [
                {
                    "name": f.name,
                    "type": f.field_type,
                    "mode": f.mode,
                    "description": f.description or "",
                }
                for f in table.schema
            ]
            logger.info("Schema retrieved", extra={"table": table_name})
            return schema
        except Exception as exc:
            logger.error("Schema retrieval failed", extra={"table": table_name, "error": str(exc)})
            raise
