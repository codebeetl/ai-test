"""LangGraph node implementations for the retail data agent."""

import json
import logging
import time
from typing import Any

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
from src.oversight.confirmation_flow import require_confirmation
from src.observability import metrics
from src.observability.progress import show as _progress, clear as _progress_clear
from src.resilience.quota_guard import quota_safe_invoke

logger = logging.getLogger(__name__)

_settings = load_settings()

# ── Session-level quota state ─────────────────────────────────────────────────
# Once daily quota is confirmed exhausted, _QUOTA_EXHAUSTED is set to the
# user-facing message so every subsequent _checked_invoke short-circuits
# immediately without retrying the LLM.
_QUOTA_EXHAUSTED: str | None = None
# Main LLM — SQL generation
_llm = ChatGoogleGenerativeAI(
    model=_settings.llm.model,
    temperature=_settings.llm.temperature,
)
# Lightweight LLM — classification and context summarisation (cheaper/faster)
_classification_llm = ChatGoogleGenerativeAI(
    model=_settings.llm.classification_model,
    temperature=0,
)  # gemini-2.5-flash-lite (config: llm.classification_model)
# Report LLM — token-capped to control verbosity and output cost
_report_llm = ChatGoogleGenerativeAI(
    model=_settings.llm.model,
    temperature=_settings.llm.temperature,
    max_output_tokens=_settings.llm.report_max_output_tokens,
)
_prefs = UserPrefsStore(
    str(_settings.memory.resolve_path(_settings.memory.user_prefs_path))
)
_DATASET = "bigquery-public-data.thelook_ecommerce"

# ── Prompt 1: SQL generation only — no persona, no prose ────────────────────
_SQL_SYSTEM_PROMPT = """You are a BigQuery SQL expert. Your ONLY job is to output a single valid BigQuery SQL query.

Dataset: `bigquery-public-data.thelook_ecommerce`

Available tables and their EXACT column names:

`bigquery-public-data.thelook_ecommerce.orders`
  order_id, user_id, status, gender, created_at, returned_at,
  shipped_at, delivered_at, num_of_item

`bigquery-public-data.thelook_ecommerce.order_items`
  id, order_id, user_id, product_id, inventory_item_id, status,
  created_at, shipped_at, delivered_at, returned_at, sale_price

`bigquery-public-data.thelook_ecommerce.products`
  id, cost, category, name, brand, retail_price, department,
  sku, distribution_center_id

`bigquery-public-data.thelook_ecommerce.users`
  id, first_name, last_name, age, gender, state, street_address,
  postal_code, city, country, latitude, longitude,
  traffic_source, created_at

Rules:
  - Output ONLY the raw SQL query. No explanation. No markdown. No code fences.
  - Do NOT write any words before or after the SQL.
  - The very first character of your response must be S (for SELECT) or W (for WITH).
  - Always use fully-qualified table names in backticks.
  - Never SELECT email, phone, phone_number, mobile, address, first_name, last_name columns.
  - Use LIMIT clauses to avoid returning more than 1000 rows.
  - For date filtering use: DATE(created_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL N DAY)
  - Revenue = SUM(oi.sale_price) joining order_items (oi) to orders (o) on order_id.
  - For country/region queries, use u.country joining users (u) to orders (o) on user_id.
  - Always close every opening parenthesis. COUNT every subquery alias.

Conversation context (use to resolve follow-up questions):
{context}

Reference examples from expert analysts:
{examples}"""

# ── Prompt 2: Report formatting only — receives data, applies persona ────────
_REPORT_SYSTEM_PROMPT = """You are a data analyst writing a report for a retail executive.
Apply this communication style: {persona}

You will receive a JSON dataset. Write a concise, insightful analysis report.
Do NOT output SQL. Do NOT output raw JSON. Write clear prose or a formatted summary."""


def _extract_text(content) -> str:
    """Safely extract string text from LLM response content."""
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ).strip()
    return str(content).strip()


def _clean_sql(raw: str) -> str:
    """Strip markdown code fences and any leading prose from LLM-generated SQL."""
    sql = raw.strip()
    # Strip code fences
    if sql.startswith("```sql"):
        sql = sql[6:]
    elif sql.startswith("```"):
        sql = sql[3:]
    if sql.endswith("```"):
        sql = sql[:-3]
    sql = sql.strip()
    # If the LLM still added prose before the SQL, find first SELECT or WITH
    upper = sql.upper()
    for keyword in ("WITH ", "SELECT ", "SELECT\n", "WITH\n"):
        idx = upper.find(keyword)
        if idx > 0:
            logger.warning("LLM added prose before SQL — trimming", extra={"trimmed_chars": idx})
            sql = sql[idx:]
            break
    return sql.strip()


def _looks_like_sql(text: str) -> bool:
    """Return True if the text appears to be SQL rather than prose."""
    upper = text.upper().lstrip()
    return upper.startswith("SELECT") or upper.startswith("WITH")


def _serialise_trios(trios: list) -> list[dict]:
    """Convert Golden Bucket trios to plain JSON-serialisable dicts."""
    return [
        {
            "question": str(t.get("question", "")),
            "sql":      str(t.get("sql", "")),
        }
        for t in trios
    ]


def _log_candidate_trio(question: str, sql: str, row_count: int) -> None:
    """Append a successful query to the candidate trios log for expert review (Req 1).

    Candidates are NOT automatically added to the Golden Bucket. A human expert
    sets promoted=true in the file, then runs scripts/promote_trios.py.
    """
    import json as _json
    from datetime import datetime, timezone

    settings = load_settings()
    path = settings.memory.resolve_path(settings.memory.candidate_trios_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "question": question,
        "sql": sql,
        "row_count": row_count,
        "promoted": False,
        "ingested": False,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(_json.dumps(entry) + "\n")
    metrics.increment("candidate_trios_logged")


def _invoke_llm(prompt_messages):
    """Invoke the LLM with back-off parameters read fresh from config.yaml."""
    settings = load_settings()
    r = settings.resilience

    @with_backoff(
        max_attempts=r.llm_max_attempts,
        min_wait=r.llm_min_wait_s,
        max_wait=r.llm_max_wait_s,
    )
    def _call(msgs):
        return _llm.invoke(msgs)

    return _call(prompt_messages)


def _invoke_classification_llm(prompt_messages):
    """Lightweight LLM for classification and summarisation (config: classification_model)."""
    settings = load_settings()
    r = settings.resilience

    @with_backoff(
        max_attempts=r.llm_max_attempts,
        min_wait=r.llm_min_wait_s,
        max_wait=r.llm_max_wait_s,
    )
    def _call(msgs):
        return _classification_llm.invoke(msgs)

    return _call(prompt_messages)


def _invoke_report_llm(prompt_messages):
    """Token-capped report LLM (config: report_max_output_tokens)."""
    settings = load_settings()
    r = settings.resilience

    @with_backoff(
        max_attempts=r.llm_max_attempts,
        min_wait=r.llm_min_wait_s,
        max_wait=r.llm_max_wait_s,
    )
    def _call(msgs):
        return _report_llm.invoke(msgs)

    return _call(prompt_messages)


def _checked_invoke(prompt_messages, *, fn=None) -> tuple[Any | None, dict | None]:
    """Call an LLM invoke function via quota_safe_invoke.

    Short-circuits immediately if daily quota was already exhausted this session.

    Args:
        prompt_messages: Formatted prompt messages.
        fn: LLM invoke function to call (default: _invoke_llm).

    Returns:
        (response, None)          on success.
        (None, quota_error_dict)  when quota/rate limit is hit.
    """
    global _QUOTA_EXHAUSTED
    if _QUOTA_EXHAUSTED:
        return None, {"quota_error": True, "kind": "daily_quota", "message": _QUOTA_EXHAUSTED}

    invoke_fn = fn or _invoke_llm
    result = quota_safe_invoke(invoke_fn, prompt_messages)
    if isinstance(result, dict) and result.get("quota_error"):
        if result.get("kind") == "daily_quota":
            _QUOTA_EXHAUSTED = result["message"]
        return None, result
    return result, None


def classify_intent(state: AgentState) -> AgentState:
    """Classify user input as ANALYSIS, DESTRUCTIVE, or OUT_OF_SCOPE (Req 2)."""
    _progress("classify")
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
    resp, quota_err = _checked_invoke(msg, fn=_invoke_classification_llm)
    if quota_err:
        state["raw_result"] = {"error": quota_err["message"]}
        return state
    label = _extract_text(resp.content).upper().strip()
    logger.info("Intent classified", extra={"label": label, "query_preview": last_message[:80]})
    metrics.increment(f"intent_{label.lower()[:20]}")

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


def _build_context(state: AgentState) -> str:
    """Build conversation context for the SQL prompt.

    Keeps the most recent context_verbatim_turns verbatim (config.yaml).
    If context_summary_enabled=true, older turns are summarised into one
    sentence via the classification LLM to save tokens.
    """
    settings = load_settings()
    verbatim_n = settings.agent.context_verbatim_turns
    summarise = settings.agent.context_summary_enabled

    all_history = state.get("messages", [])[:-1]  # exclude current message
    if not all_history:
        return "No prior context."

    recent = all_history[-verbatim_n * 2:]          # keep last N turn pairs
    older  = all_history[:-verbatim_n * 2] if len(all_history) > verbatim_n * 2 else []

    lines = []

    if older and summarise:
        older_text = " | ".join(
            f"{'User' if m.__class__.__name__ == 'HumanMessage' else 'Assistant'}: "
            f"{_extract_text(m.content)[:150]}"
            for m in older
        )
        try:
            summary_prompt = [
                ("system", "Summarise the following conversation excerpt in one concise sentence "
                           "focusing on what data was analysed and what was found."),
                ("user", older_text),
            ]
            from langchain_core.prompts import ChatPromptTemplate
            msgs = ChatPromptTemplate.from_messages(summary_prompt).format_messages(
                **{}
            )
            resp, _ = _checked_invoke(msgs, fn=_invoke_classification_llm)
            if resp:
                lines.append(f"Earlier context (summarised): {_extract_text(resp.content)}")
        except Exception as e:
            logger.debug(f"Context summarisation failed, skipping: {e}")

    for m in recent:
        role = "User" if m.__class__.__name__ == "HumanMessage" else "Assistant"
        lines.append(f"{role}: {_extract_text(m.content)[:300]}")

    return "\n".join(lines) if lines else "No prior context."


def execute_analysis(state: AgentState) -> AgentState:
    """Handle analysis queries using hybrid intelligence (Golden Bucket + BigQuery).

    Flow:
    1. Retrieve similar expert trios from the Golden Bucket (Req 1).
    2. Generate SQL only — dedicated SQL-only prompt prevents prose contamination.
    3. Validate the response looks like SQL before executing.
    4. Execute SQL with self-correction (Req 5).
    5. Format a report using a separate prompt with the current persona (Req 8).
    6. Log successful query as candidate trio for expert review (Req 1).
    """
    started = time.perf_counter()
    query_text = _extract_text(state["messages"][-1].content)

    # Step 1: retrieve similar expert trios for context
    _progress("analysis")
    similar_trios = search_golden_bucket.invoke({"query": query_text, "k": 3})

    # Step 2: generate SQL with a clean, prose-free prompt
    sql_prompt = ChatPromptTemplate.from_messages([
        ("system", _SQL_SYSTEM_PROMPT),
        ("user", "{query}"),
    ])
    # Build conversation context — verbatim recent turns + optional summary of older ones
    context_str = _build_context(state)

    sql_msg = sql_prompt.format_messages(
        query=query_text,
        examples=json.dumps(_serialise_trios(similar_trios), indent=2),
        context=context_str,
    )
    sql_resp, quota_err = _checked_invoke(sql_msg)
    if quota_err:
        state["raw_result"] = {"error": quota_err["message"]}
        metrics.increment("analysis_error")
        return state
    sql = _clean_sql(_extract_text(sql_resp.content))
    logger.info("SQL generated", extra={"sql_preview": sql[:120]})

    # Step 3: guard — if it still doesn't look like SQL, return a clear error
    if not _looks_like_sql(sql):
        logger.error("LLM returned prose instead of SQL", extra={"response_preview": sql[:200]})
        state["raw_result"] = {"error": "Could not generate a valid SQL query. Please rephrase your question."}
        state["last_sql"] = None
        metrics.increment("analysis_error")
        return state

    # Step 4: execute with self-correction
    _progress("executing")
    result = run_bigquery_query.invoke({"sql": sql})
    state["last_sql"] = sql

    if result.get("error"):
        metrics.increment("analysis_error")
        state["raw_result"] = result
        return state

    # Step 5: format a report using a separate persona-aware prompt (Req 8)
    _progress("reporting")
    # FIX (Req 8): persona loaded fresh so config.yaml changes apply without restart
    persona = load_settings().persona.to_prompt_fragment()
    rows = result.get("rows", [])
    columns = result.get("columns", [])

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows, columns=columns)
        df = mask_dataframe_pii(df)
        # Limit rows sent to report LLM — configurable via agent.report_max_rows
        max_rows = load_settings().agent.report_max_rows
        preview = df.head(max_rows).to_markdown(index=False)

        report_prompt = ChatPromptTemplate.from_messages([
            ("system", _REPORT_SYSTEM_PROMPT),
            ("user", "Question: {question}\n\nData:\n{data}"),
        ])
        report_msg = report_prompt.format_messages(
            persona=persona,
            question=query_text,
            data=preview,
        )
        report_resp, quota_err = _checked_invoke(report_msg, fn=_invoke_report_llm)
        if quota_err:
            # SQL succeeded — fall back to raw table output without a report
            logger.warning("Report generation skipped due to quota error")
            state["raw_result"] = {
                "rows": df.to_dict(orient="records"),
                "columns": list(df.columns),
                "report": None,
                "warning": quota_err["message"],
            }
            _log_candidate_trio(query_text, sql, len(rows))
            metrics.increment("analysis_success")
            metrics.record_latency(time.perf_counter() - started, "analysis_latency_s")
            return state
        state["raw_result"] = {
            "rows": df.to_dict(orient="records"),
            "columns": list(df.columns),
            "report": _extract_text(report_resp.content),
            "warning": result.get("warning"),
        }
        _log_candidate_trio(query_text, sql, len(rows))
        metrics.increment("analysis_success")
    else:
        state["raw_result"] = result

    metrics.record_latency(time.perf_counter() - started, "analysis_latency_s")
    return state


def confirmation_gate(state: AgentState) -> AgentState:
    """FIX (Req 3): Confirmation enforced HERE in the graph node, not inside the tool.

    Clears pending_destructive_op if the operator does not confirm, causing
    _route_confirmation in graph.py to route to mask_and_format instead.
    """
    _progress("destructive")
    op = state.get("pending_destructive_op") or {}
    raw_message = op.get("raw_message", "destructive operation")
    _progress_clear()  # clear before the confirmation prompt appears
    confirmed = require_confirmation(raw_message)
    if not confirmed:
        state["pending_destructive_op"] = None
        state["raw_result"] = {"message": "Deletion aborted. No changes made."}
        metrics.increment("destructive_aborted")
    else:
        metrics.increment("destructive_confirmed")
    return state


def execute_destructive(state: AgentState) -> AgentState:
    """Execute a pending destructive operation (already confirmed by confirmation_gate)."""
    op = state.get("pending_destructive_op") or {}
    raw_message = op.get("raw_message", "")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract only the client name from this delete request. Return just the name."),
        ("user", "{message}"),
    ])
    msg = prompt.format_messages(message=raw_message)
    client_resp, quota_err = _checked_invoke(msg)
    if quota_err:
        state["raw_result"] = {"message": quota_err["message"]}
        return state
    client_name = _extract_text(client_resp.content)

    result = delete_reports_by_client.invoke({"client_name": client_name})
    state["raw_result"] = {"message": result}
    return state


def mask_and_format(state: AgentState) -> AgentState:
    """Apply PII masking and user-specific formatting to the raw result (Req 2, 4.1)."""
    _progress("formatting")
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

    # If a pre-formatted report was generated, use it directly
    if raw.get("report"):
        report = mask_pii(raw["report"])
        if raw.get("warning"):
            state["final_output"] = f"Warning: {raw['warning']}\n\n{report}"
        else:
            state["final_output"] = report
        return state

    if "rows" in raw and "columns" in raw:
        import pandas as pd
        df = pd.DataFrame(raw["rows"], columns=raw["columns"])
        before_cols = set(df.columns)
        df = mask_dataframe_pii(df)
        if set(df.columns) != before_cols:
            metrics.increment("pii_columns_removed")

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
        masked = mask_pii(text)
        if masked != text:
            metrics.increment("pii_text_masked")
        state["final_output"] = masked

    _progress_clear()
    return state
