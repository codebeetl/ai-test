"""Golden Bucket search and trio storage tools exposed as LangChain @tools."""

import logging
from langchain_core.tools import tool
from src.config.settings import load_settings

logger = logging.getLogger(__name__)
_settings = load_settings()

# Lazy-initialised to avoid circular import at startup
_gb = None


def _get_gb():
    """Lazily initialise GoldenBucket with Gemini embeddings."""
    global _gb
    if _gb is None:
        from src.memory.golden_bucket import GoldenBucket
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        _gb = GoldenBucket(_settings.memory.golden_bucket_path, embedder)
    return _gb


@tool
def search_golden_bucket(query: str, k: int = 3) -> list[dict]:
    """Search the Golden Bucket for similar past analyses.

    Returns the k most similar Question → SQL → Report trios to guide
    SQL generation for new analysis questions. Returns empty list if the
    bucket has no entries yet.

    Args:
        query: Natural language question from the user.
        k: Number of similar trios to return (default 3).

    Returns:
        List of dicts with keys 'question', 'sql', 'report'.
    """
    try:
        trios = _get_gb().similarity_search(query, k)
        return [{"question": t.question, "sql": t.sql, "report": t.report} for t in trios]
    except Exception as e:
        logger.warning("Golden Bucket search failed, continuing without examples", extra={"error": str(e)})
        return []


@tool
def save_trio(question: str, sql: str, report: str) -> str:
    """Save a successful analysis as a Golden Trio for future learning.

    Called automatically after each successful BigQuery analysis to grow
    the Golden Bucket over time.

    Args:
        question: The original natural language question.
        sql: The SQL query that was executed successfully.
        report: A brief summary of the result.

    Returns:
        Confirmation message.
    """
    from dataclasses import dataclass

    @dataclass
    class Trio:
        question: str
        sql: str
        report: str

    try:
        _get_gb().add_trios([Trio(question, sql, report)])
        logger.info("Trio saved to Golden Bucket", extra={"question_preview": question[:60]})
        return "✅ Trio saved to Golden Bucket"
    except Exception as e:
        logger.warning("Failed to save trio", extra={"error": str(e)})
        return f"⚠️  Could not save trio: {e}"
