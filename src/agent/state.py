"""Defines the shared state schema that flows through every LangGraph node."""

from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Immutable-style state bag passed between graph nodes."""
    messages: Annotated[list, add_messages]
    user_id: str
    pending_destructive_op: dict[str, Any] | None
    last_sql: str | None
    retry_count: int
    raw_result: Any | None
    final_output: str | None
