"""Per-user preference store — persists manager formatting preferences.

Stores lightweight per-user settings (e.g. preferred output format: table vs
bullet points) in a local SQLite database via a JSON column.

Prototype vs Production:
  PROTOTYPE  : SQLite file at data/user_prefs.db (portable, zero-infra).
  PRODUCTION : Replace SQLitePrefsStore with a Firestore or Cloud SQL backed
               implementation. The PrefsStore abstract base class is the
               extension point — swap the concrete class in config/settings.py.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/user_prefs.db")

DEFAULT_PREFS: dict[str, Any] = {
    "output_format": "table",   # 'table' | 'bullets' | 'prose'
    "verbosity": "normal",      # 'brief' | 'normal' | 'detailed'
}


class PrefsStore(ABC):
    """Abstract preference store. Swap concrete implementation for production."""

    @abstractmethod
    def get(self, user_id: str) -> dict[str, Any]: ...

    @abstractmethod
    def set(self, user_id: str, prefs: dict[str, Any]) -> None: ...

    @abstractmethod
    def update(self, user_id: str, key: str, value: Any) -> None: ...


class SQLitePrefsStore(PrefsStore):
    """SQLite-backed preference store for prototype use.

    # PRODUCTION NOTE: Replace this class with a CloudPrefsStore that
    # reads/writes to Firestore (GCP) or DynamoDB (AWS). The interface
    # (get / set / update) remains identical — only the backend changes.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        """Initialise SQLite store, creating schema if needed.

        Args:
            db_path: Filesystem path to the SQLite database file.
        """
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._create_schema()
        logger.info("SQLitePrefsStore ready", extra={"db_path": str(db_path)})

    def _create_schema(self) -> None:
        """Create the user_prefs table if it does not already exist."""
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS user_prefs "
            "(user_id TEXT PRIMARY KEY, prefs_json TEXT NOT NULL)"
        )
        self._conn.commit()

    def get(self, user_id: str) -> dict[str, Any]:
        """Retrieve preferences for a user, returning defaults if not found."""
        row = self._conn.execute(
            "SELECT prefs_json FROM user_prefs WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row is None:
            return dict(DEFAULT_PREFS)
        return json.loads(row[0])

    def set(self, user_id: str, prefs: dict[str, Any]) -> None:
        """Overwrite all preferences for a user."""
        self._conn.execute(
            "INSERT OR REPLACE INTO user_prefs (user_id, prefs_json) VALUES (?, ?)",
            (user_id, json.dumps(prefs)),
        )
        self._conn.commit()

    def update(self, user_id: str, key: str, value: Any) -> None:
        """Update a single preference key for a user."""
        prefs = self.get(user_id)
        prefs[key] = value
        self.set(user_id, prefs)
        logger.info("User pref updated", extra={"user_id": user_id, "key": key})
