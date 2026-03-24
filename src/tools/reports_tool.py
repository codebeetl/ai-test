"""Saved Reports library tools, including GDPR-compliant delete operations."""

import sqlite3
import logging
from pathlib import Path
from typing import List
from langchain_core.tools import tool

from src.config.settings import load_settings
from src.oversight.confirmation_flow import require_confirmation

logger = logging.getLogger(__name__)
_settings = load_settings()
_DB_PATH = Path(_settings.memory.reports_path)
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
_CONN = sqlite3.connect(str(_DB_PATH))
_CUR = _CONN.cursor()
_CUR.execute("""
    CREATE TABLE IF NOT EXISTS reports (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      title TEXT NOT NULL,
      client_name TEXT,
      content TEXT NOT NULL
    )
""")
_CONN.commit()


@tool
def list_reports(client_name: str | None = None) -> List[dict]:
    """List saved reports, optionally filtered by client name.

    Args:
        client_name: Optional client identifier to filter results.

    Returns:
        List of dicts containing report id, title, and client_name.
    """
    if client_name:
        _CUR.execute(
            "SELECT id, title, client_name FROM reports WHERE client_name = ?",
            (client_name,),
        )
    else:
        _CUR.execute("SELECT id, title, client_name FROM reports")
    rows = _CUR.fetchall()
    return [{"id": r[0], "title": r[1], "client_name": r[2]} for r in rows]


@tool
def delete_reports_by_client(client_name: str) -> str:
    """Delete all saved reports that reference a given client.

    HIGH-STAKES operation: prompts the operator for an explicit confirmation
    phrase before deleting any records, satisfying GDPR delete flows.

    Args:
        client_name: Name/identifier of the client whose reports should be removed.

    Returns:
        Summary of how many reports were deleted, or an abort message.
    """
    _CUR.execute(
        "SELECT COUNT(*) FROM reports WHERE client_name = ?",
        (client_name,),
    )
    count = _CUR.fetchone()[0]
    if count == 0:
        return f"No reports found for client '{client_name}'."

    confirmed = require_confirmation(
        f"Delete {count} report(s) mentioning client '{client_name}'."
    )
    if not confirmed:
        return "Deletion aborted. No reports were removed."

    _CUR.execute("DELETE FROM reports WHERE client_name = ?", (client_name,))
    _CONN.commit()
    logger.warning("Reports deleted", extra={"client_name": client_name, "count": count})
    return f"Deleted {count} report(s) for client '{client_name}'."
