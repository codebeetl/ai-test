"""Per-user presentation preferences (tables vs bullets, etc.)."""

import sqlite3
from pathlib import Path
from typing import TypedDict, Literal

PreferenceFormat = Literal["table", "bullets"]


class UserPrefs(TypedDict, total=False):
    """User preference record.

    Fields:
        user_id: Stable identifier (e.g. email or SSO subject).
        output_format: Preferred report format: 'table' or 'bullets'.
    """
    user_id: str
    output_format: PreferenceFormat


class UserPrefsStore:
    """SQLite-backed key–value store for user preferences."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_prefs (
              user_id TEXT PRIMARY KEY,
              output_format TEXT CHECK (output_format IN ('table', 'bullets')) NOT NULL
            )
        """)
        self._conn.commit()

    def get(self, user_id: str) -> UserPrefs:
        """Fetch preferences for a user_id. Defaults to table format."""
        cur = self._conn.cursor()
        cur.execute("SELECT output_format FROM user_prefs WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        if not row:
            return {"user_id": user_id, "output_format": "table"}
        return {"user_id": user_id, "output_format": row[0]}

    def set_output_format(self, user_id: str, fmt: PreferenceFormat) -> None:
        """Persist the user's preferred output format."""
        cur = self._conn.cursor()
        cur.execute("""
            INSERT INTO user_prefs (user_id, output_format) VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET output_format = excluded.output_format
        """, (user_id, fmt))
        self._conn.commit()
