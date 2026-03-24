"""AgentState — the shared state schema flowing through every LangGraph node.

All fields are optional at construction time so nodes can be composed freely.
The `messages` field uses LangGraph's `add_messages` reducer to append-only,
ensuring conversation history is never accidentally overwritten by a node.
"""

from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """Immutable-style state bag passed between every graph node.

    Fields:
        messages: Full conversation history; appended via LangGraph reducer.
        user_id: Identifies the manager for preference + formatting lookup.
        pending_destructive_op: Populated when the intent classifier detects
            a destructive intent (e.g. delete reports). Holds action metadata
            that the confirmation_gate node will surface to the operator.
        last_sql: The most recently generated SQL string. Retained so the
            sql_self_correct loop can pass it back to the LLM for rewriting.
        retry_count: Number of SQL self-correction attempts so far. Capped at
            MAX_SQL_RETRIES to prevent runaway token spend.
        raw_result: Raw DataFrame / dict result *before* PII masking is applied.
            Never surfaced directly to the CLI layer.
        final_output: The masked, formatted response ready for CLI display.
            This is the only field the CLI rendering layer should read.
    """

    messages: Annotated[list, add_messages]
    user_id: str
    pending_destructive_op: dict[str, Any] | None
    last_sql: str | None
    retry_count: int
    raw_result: Any | None
    final_output: str | None
