"""
Central configuration loader.

Reads config.yaml and environment variables, exposing a single `settings`
object used by all other modules.
"""

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class LLMParameters(BaseModel):
    temperature: float = 0.2
    max_tokens: int = 4096


class LLMModel(BaseModel):
    name: str = "gemini-2.5-flash"


class LLMConfig(BaseModel):
    platform: str = "google"
    model: LLMModel = Field(default_factory=LLMModel)
    parameters: LLMParameters = Field(default_factory=LLMParameters)


class EmbeddingsConfig(BaseModel):
    model: str = "models/gemini-embedding-001"


class AgentConfig(BaseModel):
    max_sql_retries: int = 3
    golden_bucket_top_k: int = 3
    confirmation_keyword: str = "CONFIRM"


class PersonaConfig(BaseModel):
    tone: str = "You are a concise retail data analyst."


class BigQueryConfig(BaseModel):
    dataset: str = "bigquery-public-data.thelook_ecommerce"
    tables: list[str] = Field(default_factory=lambda: ["orders", "order_items", "products", "users"])


class MemoryConfig(BaseModel):
    """Filesystem paths for all persistent storage used by the agent."""
    base_path: str = "./data"
    golden_bucket_path: str = "golden_bucket.db"
    user_prefs_path: str = "user_prefs.json"
    reports_path: str = "reports.db"
    candidate_trios_path: str = "candidate_trios.jsonl"
    agent_log_path: str = "agent.log"

    def resolve_path(self, relative: str) -> Path:
        """Resolve a filename relative to base_path.

        Args:
            relative: Filename or relative path under base_path.

        Returns:
            Absolute Path object.
        """
        return Path(self.base_path) / relative


class GoldenBucketConfig(BaseModel):
    persist_path: str = "./golden_bucket"
    collection_name: str = "retail_examples"


class ObservabilityConfig(BaseModel):
    langsmith_project: str = "opsfleet-ai-prototype"
    log_level: str = "INFO"


class Settings(BaseModel):
    """
    Root settings object. All fields are populated from config.yaml,
    with environment variables taking precedence where applicable.
    """
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    bigquery: BigQueryConfig = Field(default_factory=BigQueryConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    golden_bucket: GoldenBucketConfig = Field(default_factory=GoldenBucketConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    # Resolved from environment at load time
    google_api_key: str = Field(default="")

    @property
    def effective_persona(self) -> str:
        """
        Returns the persona tone, preferring the AGENT_PERSONA environment
        variable so non-developers can override it without editing YAML (Req 8).
        """
        return os.getenv("AGENT_PERSONA", self.persona.tone)


@lru_cache(maxsize=1)
def load_settings(config_path: str = "config.yaml") -> Settings:
    """
    Load and cache settings from a YAML file. The cache means the file is
    read once per process — change config_path in tests to point at a fixture.
    """
    path = Path(config_path)
    raw: dict = {}

    if path.exists():
        with path.open() as f:
            raw = yaml.safe_load(f) or {}

    settings = Settings.model_validate(raw)
    settings = settings.model_copy(
        update={"google_api_key": os.getenv("GOOGLE_API_KEY", "")}
    )

    return settings


# Module-level singleton — import this in other modules
settings = load_settings()
