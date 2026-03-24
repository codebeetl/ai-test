"""Golden Bucket storage — Trio retrieval and incremental updates.

Small-scale: FAISS index + SQLite. Rebuilds the FAISS index from the
database on each add_trios call. Suitable for prototype scale (<10k trios).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
import sqlite3
import logging

import faiss
import numpy as np
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


@dataclass
class Trio:
    """A Golden Bucket record capturing a past expert analysis."""
    question: str
    sql: str
    report: str


class GoldenBucket:
    """Vector-searchable Golden Bucket over Trio records."""

    DEFAULT_DIM = 768  # Gemini embedding-001 dimension

    def __init__(self, path: str, embedder: Embeddings | None) -> None:
        self._db_path = Path(path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._embedder = embedder
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trios (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              question TEXT NOT NULL,
              sql TEXT NOT NULL,
              report TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def add_trios(self, trios: Iterable[Trio]) -> None:
        """Insert new Trio records into the database."""
        cur = self._conn.cursor()
        for trio in trios:
            cur.execute(
                "INSERT INTO trios (question, sql, report) VALUES (?, ?, ?)",
                (trio.question, trio.sql, trio.report),
            )
        self._conn.commit()
        logger.info("Trios added to Golden Bucket")

    def similarity_search(self, query: str, k: int = 3) -> List[Trio]:
        """Find the k most similar trios to a user question using FAISS."""
        if self._embedder is None:
            logger.warning("No embedder configured, skipping similarity search")
            return []

        cur = self._conn.cursor()
        cur.execute("SELECT id, question, sql, report FROM trios")
        rows = cur.fetchall()
        if not rows:
            return []

        questions = [q for _, q, _, _ in rows]
        vectors = self._embedder.embed_documents(questions)
        query_vec = self._embedder.embed_query(query)
        dim = len(query_vec)

        index = faiss.IndexFlatL2(dim)
        index.add(np.array(vectors, dtype="float32"))
        _, indices = index.search(np.array([query_vec], dtype="float32"), min(k, len(rows)))

        result: List[Trio] = []
        for idx in indices[0]:
            if 0 <= idx < len(rows):
                _, q, sql, report = rows[idx]
                result.append(Trio(question=q, sql=sql, report=report))
        return result
