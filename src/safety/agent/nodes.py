"""LangGraph node implementations for the retail data agent."""

import json
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from src.agent.state import AgentState
from src.config.settings import load_settings
from src.tools.query_tool import run_bigquery_query
from src.tools.reports_tool import delete_reports_by_client
from src.tools.golden_bucket_tool import search_golden_bucket
from src.safety.pii_masker import mask_pii, mask_dataframe_pii
from src.memory.user_prefs import UserPrefsStore
from src.resilience.retry import with_backoff

logger = logging.getLogger(__name__)

_settings = load_settings()
_llm = ChatGoogleGenerativeAI(
    model=_settings.llm.model,
    temperature=_settings.llm.temperature,
)
_prefs = UserPrefsStore(
    str(_settings.memory.resolve_path(_settings.memory.user_prefs_path))
)
_PERSONA = _settings.persona.to_prompt_fragment()
_DATASET = "bigquery-public-data.thelook_ecommerce"

_SQL_SYSTEM_PROMPT = (
    "You are a BigQuery SQL expert.\n"
    "You MUST use fully-qualified table names in EVERY query.\n"
    f"Dataset: `{_DATASET}`\n\n"
    "Available tables — use EXACTLY these references:\n"
    f"  `{_DATASET}.orders`\n"
    f"  `{_DATASET}.order_items`\n"
    f"  `{_DATASET}.products`\n"
    f"  `{_DATASET}.users`\n\n"
    "Rules:\n"
    "  - Never use unqualified names like orders, your_dataset.orders, or thelook.orders\n"
    "  - Always wrap table references in backticks\n"
    "  - Return ONLY valid BigQuery SQL. No explanation, no markdown fences.\n\n"
    "When formatting the final report, apply this persona:\n"
    "{persona}\n\n"
    "Similar expert analyses for reference:\n{examples}"
)


def _extract_text(content) -> str:
    """Safely extract string text from LLM response content."""
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ).strip()
    return str(content).strip()


def _clean_sql(raw: str) -> str:
    """Strip markdown code fences from LLM-generated SQL."""
    sql = raw.strip()
    if sql.startswith("```sql"):
        sql = sql[6:]
    elif sql.startswith("```"):
        sql = sql[3:]
    if sql.endswith("```"):
        sql = sql[:-3]
    return sql.strip()


def _serialise_trios(trios: list) -> list[dict]:
    """Convert Golden Bucket trios to plain JSON-serialisable dicts."""
    return [
        {
            "question": str(t.get("question", "")),
            "sql":      str(t.get("sql", "")),
            "report":   str(t.get("report", "")),
        }
        for t in trios
    ]


def _log_candidate_trio(question: str, sql: str, row_count: int) -> None:
    """Append a successful query to the candidate trios log for expert review (Req 4.2)."""
    import json as _json
    from pathlib import Path
    from datetime import datetime, timezone

    path = _settings.memory.resolve_path(_settings.memory.candidate_trios_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "question": question,
        "sql": sql,
        "row_count": row_count,
        "promoted": False,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(_json.dumps(entry) + "\n")


@with_backoff(max_attempts=5, min_wait=5, max_wait=60)
def _invoke_llm(prompt_messages):
    """Invoke the LLM with exponential back-off on rate limit errors."""
    return _llm.invoke(prompt_messages)


def classify_intent(state: AgentState) -> AgentState:
    """Classify user input as ANALYSIS, DESTRUCTIVE, or OUT_OF_SCOPE (Req 2)."""
    last_message = _extract_text(state["messages"][-1].content)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a classification model for a retail data analysis assistant.\n"
         "Classify the user message as exactly one of:\n"
         "  ANALYSIS     - a question about sales, orders, products, customers, or performance\n"
         "  DESTRUCTIVE  - a request to delete or remove reports\n"
         "  OUT_OF_SCOPE - anything else (greetings, small talk, unrelated questions)\n"
         "Reply with one word only."),
        ("user", "{query}"),
    ])
    msg = prompt.format_messages(query=last_message)
    resp = _invoke_llm(msg)
    label = _extract_text(resp.content).upper().strip()
    logger.info("Intent classified", extra={"label": label, "query_preview": last_message[:80]})

    if "DESTRUCTIVE" in label and "delete" in last_message.lower():
        state["pending_destructive_op"] = {"raw_message": last_message}
        state["raw_result"] = None
    elif "OUT_OF_SCOPE" in label:
        state["pending_destructive_op"] = None
        state["raw_result"] = {"out_of_scope": True}
    else:
        state["pending_destructive_op"] = None
        state["raw_result"] = None

    return state


def execute_analysis(state: AgentState) -> AgentState:
    """Handle analysis queries using hybrid intelligence (Golden Bucket + BigQuery).

    Flow:
    1. Retrieve similar expert trios from the Golden Bucket (Req 1).
    2. Inject persona tone/style into the system prompt (Req 8).
    3. Generate SQL with fully-qualified table names.
    4. Execute SQL with self-correction (Req 5).
    5. Log successful query as candidate trio for expert review (Req 4.2).
    """
    query_text = _extract_text(state["messages"][-1].content)

    similar_trios = search_golden_bucket.invoke({"query": query_text, "k": 3})

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SQL_SYSTEM_PROMPT),
        ("user", "{query}"),
    ])
    msg = prompt.format_messages(
        query=query_text,
        persona=_PERSONA,
        examples=json.dumps(_serialise_trios(similar_trios), indent=2),
    )
    sql_resp = _invoke_llm(msg)
    sql = _clean_sql(_extract_text(sql_resp.content))
    logger.info("SQL generated", extra={"sql_preview": sql[:120]})

    result = run_bigquery_query.invoke({"sql": sql})
    state["raw_result"] = result
    state["last_sql"] = sql

    if result.get("rows"):
        _log_candidate_trio(query_text, sql, len(result["rows"]))

    return state


def confirmation_gate(state: AgentState) -> AgentState:
    """Routing node for destructive operations — confirmation enforced downstream."""
    return state


def execute_destructive(state: AgentState) -> AgentState:
    """Execute a pending destructive operation after operator confirmation (Req 3)."""
    op = state.get("pending_destructive_op") or {}
    raw_message = op.get("raw_message", "")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract only the client name from this delete request. Return just the name."),
        ("user", "{message}"),
    ])
    msg = prompt.format_messages(message=raw_message)
    client_resp = _invoke_llm(msg)
    client_name = _extract_text(client_resp.content)

    result = delete_reports_by_client.invoke({"client_name": client_name})
    state["raw_result"] = {"message": result}
    return state


def mask_and_format(state: AgentState) -> AgentState:
    """Apply PII masking and user-specific formatting to the raw result (Req 2, 4.1)."""
    user_id = state.get("user_id", "anonymous")
    prefs = _prefs.get(user_id)
    raw = state.get("raw_result") or {}

    if raw.get("out_of_scope"):
        state["final_output"] = (
            "I can only help with retail data analysis questions, such as:\n"
            "  - Sales and revenue trends\n"
            "  - Product performance\n"
            "  - Customer behaviour and order history\n"
            "  - Inventory and returns\n"
            "  - Managing saved reports\n\n"
            "Please ask a question about the data."
        )
        return state

    if raw.get("error"):
        state["final_output"] = (
            f"I was unable to complete that query: {raw['error']}\n"
            "Please try rephrasing your question."
        )
        return state

    if "rows" in raw and "columns" in raw:
        import pandas as pd
        df = pd.DataFrame(raw["rows"], columns=raw["columns"])
        df = mask_dataframe_pii(df)

        if df.empty:
            state["final_output"] = "No results found for your query."
            return state

        if prefs["output_format"] == "table":
            state["final_output"] = df.to_markdown(index=False)
        else:
            state["final_output"] = "\n".join(
                f"- {', '.join(f'{k}: {v}' for k, v in row.items())}"
                for row in df.to_dict(orient="records")
            )

        if raw.get("warning"):
            state["final_output"] = f"Warning: {raw['warning']}\n\n" + state["final_output"]
    else:
        text = str(raw.get("message", raw))
        state["final_output"] = mask_pii(text)

    return state
