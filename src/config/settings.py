"""Application configuration layer, loaded from config.yaml."""

from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field


class LLMSettings(BaseModel):
    """Configuration for the primary LLM provider."""
    provider: str = Field(default="gemini")
    model: str = Field(default="gemini-1.5-flash")
    temperature: float = Field(default=0.2)
    max_tokens: int = Field(default=1024)


class BigQuerySettings(BaseModel):
    """Configuration for BigQuery access."""
    project_id: str | None = None
    dataset_id: str = "bigquery-public-data.thelook_ecommerce"


class MemoryPaths(BaseModel):
    """File paths for local SQLite-backed persistence."""
    golden_bucket_path: str = "data/golden_trios.sqlite"
    user_prefs_path: str = "data/user_prefs.sqlite"
    reports_path: str = "data/saved_reports.sqlite"


class AppSettings(BaseModel):
    """Top-level application settings object."""
    llm: LLMSettings
    bigquery: BigQuerySettings
    memory: MemoryPaths


def load_settings(config_path: str = "config.yaml") -> AppSettings:
    """Load application settings from a YAML config file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed AppSettings instance.
    """
    path = Path(config_path)
    data: dict[str, Any] = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AppSettings(
        llm=LLMSettings(**data.get("llm", {})),
        bigquery=BigQuerySettings(**data.get("bigquery", {})),
        memory=MemoryPaths(**data.get("memory", {})),
    )
