"""Saved Reports library — CRUD store for persisted agent-generated reports.

Stores reports as JSON records in a local SQLite database for the prototype.

Prototype vs Production:
  PROTOTYPE  : SQLite at data/reports.db — portable, zero-infra.
  PRODUCTION : Replace SQLiteReportsStore with a GCSReportsStore that writes
               JSON objects to a Google Cloud Storage bucket.
               Each report is a separate GCS object keyed by report_id.
               Deletion uses the GCS delete object API, which is auditable via
               Cloud Audit Logs (satisfying GDPR right-to-erasure traceability).

               # PRODUCTION SWAP:
               # from src.memory.gcs_reports_store import GCSReportsStore
               # store = GCSReportsStore(bucket_name=settings.reports_bucket)

Destructive operations:
  delete_reports_by_client() is the ONLY mutation method. It MUST be called
  exclusively through the oversight.confirmation_flow gate. Direct calls
  from outside the graph are considered a bug.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/reports.db")


class SQLiteReportsStore:
    """SQLite-backed reports store for prototype use.

    # PRODUCTION NOTE: Swap to GCSReportsStore. GCS gives you:
    #   - Object versioning (soft-delete / recovery window)
    #   - Cloud Audit Logs for GDPR deletion audit trails
    #   - IAM-controlled access per bucket/prefix
    #   - No practical storage limit
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        """Initialise the SQLite reports store.

        Args:
            db_path: Path to the SQLite database file.
        """
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._create_schema()
        logger.info("SQLiteReportsStore ready", extra={"db_path": str(db_path)})

    def _create_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                report_id   TEXT PRIMARY KEY,
                title       TEXT NOT NULL,
                content     TEXT NOT NULL,
                metadata    TEXT NOT NULL DEFAULT '{}',
                created_at  TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def save_report(
        self,
        title: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist a new report and return its generated ID.

        Args:
            title: Short descriptive title for the report.
            content: Full report body (markdown or plain text).
            metadata: Optional dict of tags (e.g. {'client': 'X', 'region': 'EMEA'}).

        Returns:
            The UUID assigned to the new report.
        """
        report_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO reports (report_id, title, content, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                report_id,
                title,
                content,
                json.dumps(metadata or {}),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        logger.info("Report saved", extra={"report_id": report_id, "title": title})
        return report_id

    def list_reports(self) -> list[dict[str, Any]]:
        """Return all saved reports (id, title, created_at) without full content."""
        rows = self._conn.execute(
            "SELECT report_id, title, created_at FROM reports ORDER BY created_at DESC"
        ).fetchall()
        return [{"report_id": r[0], "title": r[1], "created_at": r[2]} for r in rows]

    def find_reports_mentioning(self, client_name: str) -> list[dict[str, Any]]:
        """Find reports whose title or content mentions a given client name.

        Used by the delete flow to preview scope before the confirmation gate.

        Args:
            client_name: Plain text name to search for (case-insensitive).

        Returns:
            List of matching report dicts with report_id, title, created_at.
        """
        pattern = f"%{client_name}%"
        rows = self._conn.execute(
            "SELECT report_id, title, created_at FROM reports "
            "WHERE title LIKE ? OR content LIKE ?",
            (pattern, pattern),
        ).fetchall()
        return [{"report_id": r[0], "title": r[1], "created_at": r[2]} for r in rows]

    def delete_reports_by_client(self, client_name: str) -> int:
        """Permanently delete all reports mentioning a client name.

        WARNING: This is a destructive, irreversible operation.
        It MUST only be called after require_confirmation() returns True.
        Caller is responsible for the confirmation gate.

        Args:
            client_name: Client name to match against title and content.

        Returns:
            Number of reports deleted.
        """
        pattern = f"%{client_name}%"
        cursor = self._conn.execute(
            "DELETE FROM reports WHERE title LIKE ? OR content LIKE ?",
            (pattern, pattern),
        )
        self._conn.commit()
        deleted = cursor.rowcount
        logger.warning(
            "Reports deleted",
            extra={"client_name": client_name, "count": deleted},
        )
        return deleted
