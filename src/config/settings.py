"""Application configuration layer, loaded from config.yaml and persona.yaml."""

from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field


class LLMSettings(BaseModel):
    provider: str = Field(default="gemini")
    model: str = Field(default="gemini-2.5-flash")
    temperature: float = Field(default=0.2)
    max_tokens: int = Field(default=8192)


class BigQuerySettings(BaseModel):
    project_id: str | None = None
    dataset_id: str = "bigquery-public-data.thelook_ecommerce"


class MemoryPaths(BaseModel):
    golden_bucket_path: str = "data/golden_trios.sqlite"
    user_prefs_path: str = "data/user_prefs.sqlite"
    reports_path: str = "data/saved_reports.sqlite"
    candidate_trios_path: str = "data/candidate_trios.jsonl"

    def resolve_path(self, relative: str) -> Path:
        """Resolve a data path relative to the project root.

        Guarantees init_data.py, main.py, and all nodes always read and write
        to the same absolute path regardless of cwd.
        """
        project_root = Path(__file__).parent.parent.parent
        return project_root / relative


class PersonaSettings(BaseModel):
    """Loaded from persona.yaml — editable by non-developers without redeployment (Req 8)."""
    tone: str = "professional and concise"
    style_hints: list[str] = Field(default_factory=list)

    def to_prompt_fragment(self) -> str:
        """Render persona as a system prompt fragment."""
        hints = "\n".join(f"  - {h}" for h in self.style_hints)
        return f"Tone: {self.tone}\nStyle guidelines:\n{hints}"


class AppSettings(BaseModel):
    llm: LLMSettings
    bigquery: BigQuerySettings
    memory: MemoryPaths
    persona: PersonaSettings


def load_settings(
    config_path: str = "config.yaml",
    persona_path: str = "src/config/persona.yaml",
) -> AppSettings:
    """Load application settings from YAML config files.

    Both files are read at runtime on every agent startup, so changes to
    persona.yaml take effect without redeployment (satisfies Req 8).
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
        persona=PersonaSettings(**persona_data),
    )
