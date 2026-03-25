"""Golden Bucket search tool with TTL cache to avoid repeat embedding calls."""

import hashlib
import logging
import time
from langchain_core.tools import tool
from src.config.settings import load_settings

logger = logging.getLogger(__name__)

_gb = None
# TTL cache: {query_hash: (timestamp, result)}
_cache: dict[str, tuple[float, list]] = {}


def _get_gb():
    """Lazily initialise GoldenBucket using embedding_model from config.yaml."""
    global _gb
    if _gb is None:
        from src.memory.golden_bucket import GoldenBucket
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        settings = load_settings()
        embedder = GoogleGenerativeAIEmbeddings(model=settings.llm.embedding_model)
        _gb = GoldenBucket(settings.memory.golden_bucket_path, embedder)
    return _gb


def _cache_key(query: str, k: int) -> str:
    return hashlib.md5(f"{query}:{k}".encode()).hexdigest()


def _get_cached(query: str, k: int) -> list | None:
    """Return cached result if still within TTL, else None."""
    settings = load_settings()
    ttl = settings.agent.golden_bucket_cache_ttl_s
    if ttl <= 0:
        return None  # caching disabled
    key = _cache_key(query, k)
    if key in _cache:
        ts, result = _cache[key]
        if time.time() - ts < ttl:
            logger.debug("Golden Bucket cache hit", extra={"query_preview": query[:60]})
            return result
        del _cache[key]
    return None


def _set_cached(query: str, k: int, result: list) -> None:
    """Store result in cache, evicting oldest entries beyond max size."""
    settings = load_settings()
    ttl = settings.agent.golden_bucket_cache_ttl_s
    if ttl <= 0:
        return
    max_size = settings.agent.golden_bucket_cache_size
    if len(_cache) >= max_size:
        # Evict oldest entry
        oldest = min(_cache, key=lambda k: _cache[k][0])
        del _cache[oldest]
    _cache[_cache_key(query, k)] = (time.time(), result)


@tool
def search_golden_bucket(query: str, k: int = 3) -> list[dict]:
    """Search the Golden Bucket for similar past analyses.

    Results are cached for golden_bucket_cache_ttl_s seconds (config.yaml)
    to avoid repeat embedding API calls for the same or similar queries.

    Args:
        query: Natural language question from the user.
        k: Number of similar trios to return (default 3).

    Returns:
        List of dicts with keys 'question', 'sql', 'report'.
    """
    cached = _get_cached(query, k)
    if cached is not None:
        return cached

    try:
        trios = _get_gb().similarity_search(query, k)
        result = [{"question": t.question, "sql": t.sql, "report": t.report} for t in trios]
        _set_cached(query, k, result)
        return result
    except Exception as e:
        logger.warning(
            "Golden Bucket search failed, continuing without examples",
            extra={"error": str(e)},
        )
        return []


@tool
def save_trio(question: str, sql: str, report: str) -> str:
    """Save a successful analysis as a Golden Trio for future learning."""
    from dataclasses import dataclass

    @dataclass
    class Trio:
        question: str
        sql: str
        report: str

    try:
        _get_gb().add_trios([Trio(question, sql, report)])
        # Invalidate cache entries that might be stale after adding new trio
        _cache.clear()
        logger.info("Trio saved to Golden Bucket", extra={"question_preview": question[:60]})
        return "Trio saved to Golden Bucket"
    except Exception as e:
        logger.warning("Failed to save trio", extra={"error": str(e)})
        return f"Could not save trio: {e}"
