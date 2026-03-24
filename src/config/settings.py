"""Application configuration layer — config.yaml is the single source of truth.

All key variables (model names, embedding model, confirmation phrase, PII columns,
retry parameters, default user) are defined in config.yaml and surfaced here as
typed Pydantic models.  Nothing in the codebase should hardcode these values.
"""

from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field


class LLMSettings(BaseModel):
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    temperature: float = 0.2
    max_tokens: int = 8192
    correction_model: str = "gemini-2.5-flash"
    embedding_model: str = "models/embedding-001"


class BigQuerySettings(BaseModel):
    project_id: str | None = None
    dataset_id: str = "bigquery-public-data.thelook_ecommerce"


class MemoryPaths(BaseModel):
    golden_bucket_path: str = "data/golden_trios.sqlite"
    user_prefs_path: str = "data/user_prefs.sqlite"
    reports_path: str = "data/saved_reports.sqlite"
    candidate_trios_path: str = "data/candidate_trios.jsonl"

    def resolve_path(self, relative: str) -> Path:
        """Resolve a data path relative to the project root."""
        project_root = Path(__file__).parent.parent.parent
        return project_root / relative


class SafetySettings(BaseModel):
    confirm_phrase: str = "YES DELETE"
    default_user_id: str = "manager_a"
    pii_columns: list[str] = Field(
        default_factory=lambda: ["email", "phone", "phone_number", "mobile", "address"]
    )


class ResilienceSettings(BaseModel):
    llm_max_attempts: int = 5
    llm_min_wait_s: float = 5.0
    llm_max_wait_s: float = 60.0
    bq_max_attempts: int = 3
    bq_min_wait_s: float = 2.0
    bq_max_wait_s: float = 15.0
    sql_max_retries: int = 2


class PersonaSettings(BaseModel):
    """Loaded from persona.yaml — editable by non-developers without redeployment (Req 8)."""
    tone: str = "professional and concise"
    style_hints: list[str] = Field(default_factory=list)

    def to_prompt_fragment(self) -> str:
        hints = "\n".join(f"  - {h}" for h in self.style_hints)
        return f"Tone: {self.tone}\nStyle guidelines:\n{hints}"


class AppSettings(BaseModel):
    llm: LLMSettings
    bigquery: BigQuerySettings
    memory: MemoryPaths
    safety: SafetySettings
    resilience: ResilienceSettings
    persona: PersonaSettings


def load_settings(
    config_path: str = "config.yaml",
    persona_path: str = "src/config/persona.yaml",
) -> AppSettings:
    """Load application settings from config.yaml (and persona.yaml for tone).

    Both files are read at call time so changes apply without restarting
    the process (satisfies Req 8 for persona; also allows live tuning of
    model, retry parameters, confirm phrase, etc. via config.yaml).
    """
    data: dict[str, Any] = {}
    if Path(config_path).exists():
        data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}

    persona_data: dict[str, Any] = {}
    if Path(persona_path).exists():
        persona_data = yaml.safe_load(Path(persona_path).read_text(encoding="utf-8")) or {}

    return AppSettings(
        llm=LLMSettings(**data.get("llm", {})),
        bigquery=BigQuerySettings(**data.get("bigquery", {})),
        memory=MemoryPaths(**data.get("memory", {})),
        safety=SafetySettings(**data.get("safety", {})),
        resilience=ResilienceSettings(**data.get("resilience", {})),
        persona=PersonaSettings(**persona_data),
    )
