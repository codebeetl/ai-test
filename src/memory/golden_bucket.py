"""Golden Bucket — vector store of expert Question→SQL→Report trios.

The Golden Bucket stores historical 'Trios' created by human analysts:
  { question: str, sql: str, report: str }

The agent uses similarity search against this store to retrieve analogous
past analyses, helping it interpret ambiguous questions and generate
higher-quality SQL aligned with how analysts previously solved similar problems.

Prototype vs Production:
  PROTOTYPE  : ChromaDB running locally with a persistent directory at
               data/golden_bucket/. Uses sentence-transformers for embeddings.
  PRODUCTION : Replace ChromaDB with Vertex AI Matching Engine (Vector Search).
               The GoldenBucketStore abstract base class is the extension point.
               Production embeddings would use text-embedding-004 via Vertex AI.
               See: https://cloud.google.com/vertex-ai/docs/vector-search/overview

               # PRODUCTION SWAP:
               # from src.memory.vertex_golden_bucket import VertexGoldenBucketStore
               # store = VertexGoldenBucketStore(index_endpoint=..., deployed_index_id=...)

Learning Loop:
  New trios are added via add_trio() after successful human-verified sessions.
  The store is persistent (ChromaDB writes to disk), so learning accumulates
  across agent restarts without any additional infrastructure.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = Path("data/golden_bucket")
DEFAULT_COLLECTION = "golden_trios"
DEFAULT_TOP_K = 3


class GoldenBucketStore(ABC):
    """Abstract golden bucket store. Swap concrete implementation for production."""

    @abstractmethod
    def search(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]: ...

    @abstractmethod
    def add_trio(self, question: str, sql: str, report: str) -> None: ...


class ChromaGoldenBucketStore(GoldenBucketStore):
    """ChromaDB-backed golden bucket for prototype use.

    # PRODUCTION NOTE: Vertex AI Matching Engine replaces this.
    # ChromaDB is appropriate for local dev and demos but does not scale
    # to millions of trios or support fine-grained IAM access control.
    # Swap to VertexGoldenBucketStore (to be implemented) in settings.py
    # when moving to production.
    """

    def __init__(
        self,
        persist_dir: Path = DEFAULT_CHROMA_PATH,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        """Initialise the ChromaDB client and collection.

        Args:
            persist_dir: Directory for ChromaDB persistence (survives restarts).
            collection_name: Name of the Chroma collection for golden trios.
        """
        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError as exc:
            raise ImportError(
                "chromadb is required: pip install chromadb sentence-transformers"
            ) from exc

        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))

        # sentence-transformers/all-MiniLM-L6-v2 — lightweight, no API key needed.
        # PRODUCTION: Replace with VertexAIEmbeddingFunction using text-embedding-004.
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name, embedding_function=ef
        )
        logger.info(
            "ChromaGoldenBucketStore ready",
            extra={"persist_dir": str(persist_dir), "doc_count": self._collection.count()},
        )

    def search(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """Find the most semantically similar past trios for a given question.

        Args:
            question: The user's natural language question.
            top_k: Number of trios to retrieve.

        Returns:
            List of dicts with keys: question, sql, report, distance.
            Returns empty list if the collection is empty.
        """
        if self._collection.count() == 0:
            logger.info("Golden bucket is empty — no analogues found")
            return []

        results = self._collection.query(
            query_texts=[question],
            n_results=min(top_k, self._collection.count()),
            include=["metadatas", "distances"],
        )
        trios = []
        for meta, dist in zip(
            results["metadatas"][0], results["distances"][0]
        ):
            trios.append({**meta, "distance": dist})
        logger.info("Golden bucket search complete", extra={"results": len(trios)})
        return trios

    def add_trio(
        self, question: str, sql: str, report: str, trio_id: str | None = None
    ) -> None:
        """Persist a new expert trio to the golden bucket.

        Called by the learning loop after a session is marked as high-quality
        by a human reviewer or automated quality gate.

        Args:
            question: The original natural language question.
            sql: The validated SQL that correctly answers the question.
            report: The analyst-quality report generated from the SQL results.
            trio_id: Optional stable ID; auto-generated if not provided.
        """
        import uuid

        doc_id = trio_id or str(uuid.uuid4())
        self._collection.add(
            ids=[doc_id],
            documents=[question],
            metadatas=[{"question": question, "sql": sql, "report": report}],
        )
        logger.info("Trio added to golden bucket", extra={"id": doc_id})
