"""LangGraph node implementations for the retail data agent."""

import json
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from src.agent.state import AgentState
from src.config.settings import load_settings
from src.tools.query_tool import run_bigquery_query
from src.tools.reports_tool import delete_reports_by_client
from src.tools.golden_bucket_tool import search_golden_bucket, save_trio
from src.safety.pii_masker import mask_pii, mask_dataframe_pii
from src.memory.user_prefs import UserPrefsStore

logger = logging.getLogger(__name__)
_settings = load_settings()
_llm = ChatGoogleGenerativeAI(
    model=_settings.llm.model,
    temperature=_settings.llm.temperature,
    thinking_budget=512,          # optional: cap thinking tokens to control cost
)
_prefs = UserPrefsStore(_settings.memory.user_prefs_path)


def classify_intent(state: AgentState) -> AgentState:
    """Classify user input as analysis or destructive action.

    Uses a lightweight LLM prompt to detect GDPR-style delete intents.
    Sets pending_destructive_op in state if a destructive intent is found.
    """
    last_message = state["messages"][-1]["content"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify this intent as exactly one word: ANALYSIS or DESTRUCTIVE."),
        ("user", "{query}"),
    ])
    msg = prompt.format_messages(query=last_message)
    resp = _llm.invoke(msg)
    label = resp.content.strip().upper()
    logger.info("Intent classified", extra={"label": label})

    if "DESTRUCTIVE" in label and "delete" in last_message.lower():
        state["pending_destructive_op"] = {"raw_message": last_message}
    else:
        state["pending_destructive_op"] = None
    return state


def execute_analysis(state: AgentState) -> AgentState:
    """Handle analysis queries using hybrid intelligence (Golden Bucket + BigQuery).

    1. Retrieve similar past trios from the Golden Bucket.
    2. Generate SQL using those trios as worked examples.
    3. Execute SQL with self-correction via run_bigquery_query.
    4. Save successful result back to Golden Bucket.
    """
    query_text = state["messages"][-1]["content"]

    # Step 1: Retrieve similar past analyses
    similar_trios = search_golden_bucket.invoke({"query": query_text, "k": 3})

    # Step 2: Generate SQL using Golden Bucket examples
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a BigQuery SQL expert for thelook_ecommerce dataset.\n"
         "Tables: orders, order_items, products, users.\n"
         "Return ONLY valid BigQuery SQL. No explanation, no markdown.\n"
         "Similar past analyses for reference:\n{examples}"),
        ("user", "{query}"),
    ])
    msg = prompt.format_messages(
        query=query_text,
        examples=json.dumps(similar_trios, indent=2),
    )
    sql_resp = _llm.invoke(msg)
    sql = sql_resp.content.strip().strip("```sql").strip("```").strip()
    logger.info("SQL generated", extra={"sql_preview": sql[:120]})

    # Step 3: Execute with self-correction
    result = run_bigquery_query.invoke({"sql": sql})
    state["raw_result"] = result
    state["last_sql"] = sql

    # Step 4: Save successful trio for future learning
    if result.get("rows"):
        save_trio.invoke({
            "question": query_text,
            "sql": sql,
            "report": f"Query returned {len(result['rows'])} rows.",
        })

    return state


def confirmation_gate(state: AgentState) -> AgentState:
    """Routing node for destructive operations.

    Exists to make the graph flow explicit. The actual confirmation prompt
    is handled inside execute_destructive via the oversight module.
    """
    return state


def execute_destructive(state: AgentState) -> AgentState:
    """Execute a pending destructive operation after confirmation.

    Extracts client_name from the pending op and calls the delete tool,
    which internally enforces the YES DELETE confirmation gate.
    """
    op = state.get("pending_destructive_op") or {}
    raw_message = op.get("raw_message", "")

    # Extract client name from message
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract only the client name from this delete request. Return just the name."),
        ("user", "{message}"),
    ])
    msg = prompt.format_messages(message=raw_message)
    client_resp = _llm.invoke(msg)
    client_name = client_resp.content.strip()

    result = delete_reports_by_client.invoke({"client_name": client_name})
    state["raw_result"] = {"message": result}
    return state


def mask_and_format(state: AgentState) -> AgentState:
    """Apply PII masking and user-specific formatting to the raw result.

    This is the single choke point through which all output passes before
    reaching the CLI. PII masking is always applied regardless of result type.
    """
    user_id = state.get("user_id", "anonymous")
    prefs = _prefs.get(user_id)
    raw = state.get("raw_result") or {}

    if raw.get("error"):
        state["final_output"] = (
            f"⚠️  I was unable to complete that query: {raw['error']}\n"
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
            state["final_output"] = f"⚠️  {raw['warning']}\n\n" + state["final_output"]
    else:
        text = str(raw.get("message", raw))
        state["final_output"] = mask_pii(text)

    return state
