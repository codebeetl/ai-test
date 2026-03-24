"""Defines the shared state schema that flows through every LangGraph node."""

from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Immutable-style state bag passed between graph nodes.

    Fields:
        messages: Full conversation history, appended via LangGraph reducer.
        user_id: Identifies the manager for preference lookup.
        pending_destructive_op: Holds a destructive action awaiting confirmation.
        last_sql: The most recently generated SQL for self-correction retries.
        retry_count: Tracks SQL self-correction attempts to cap at MAX_SQL_RETRIES.
        raw_result: Raw DataFrame/dict result before PII masking is applied.
        final_output: The masked, formatted response ready for CLI display.
    """
    messages: Annotated[list, add_messages]
    user_id: str
    pending_destructive_op: dict[str, Any] | None
    last_sql: str | None
    retry_count: int
    raw_result: Any | None
    final_output: str | None
