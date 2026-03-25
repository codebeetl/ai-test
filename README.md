# Retail Data Agent

> ⚠️ **AI Assistance Disclosure**: This prototype was scaffolded with the
> assistance of AI tooling (Perplexity + GitHub Copilot). All architecture
> decisions, code review, and testing were performed by the author.

An internal AI data analysis agent for non-technical retail executives,
built on LangGraph + BigQuery. Supports natural language queries over
`bigquery-public-data.thelook_ecommerce`.

---

## Features

| Capability | Detail |
|---|---|
| Natural language → SQL | Three-model Gemini pipeline: classify → SQL generate → report |
| Model cascading | Lightweight `gemini-2.5-flash-lite` for classification, correction, and context summarisation; `gemini-2.5-flash` for SQL and reports |
| PII masking | Column drop at data layer + regex at output layer; column list in `config.yaml` |
| Destructive op confirmation | Client name extracted at classification; operator must type exact phrase; database/DDL operations rejected before confirmation |
| SQL self-correction | LLM rewrites failed SQL with full schema context; up to `sql_max_retries` attempts |
| Golden Bucket | Expert-curated Q→SQL→Report trios; TTL-cached similarity search to reduce embedding API calls |
| Conversation context | Last N turns kept verbatim; older turns summarised by `flash-lite` to save tokens |
| Per-user output format | `table` or `bullets` preference stored per user |
| Quota guard | Distinguishes daily exhaustion from TPM/RPM limits; shows exact reset time; session latch prevents retrying after daily quota hit |
| Startup quota check | Probes the API before the main loop; exits cleanly with reset time if daily quota is already exhausted |
| Progress feedback | Single-line overwriting stage indicators; internal warnings go to log only |
| Metrics snapshot | Session counters and latencies written to `data/metrics_snapshot.json` on exit |
| Structured logging | All activity written as JSON to `data/agent.log`; console is fully silent except final output |

---

## Models

Three separate Gemini model instances are used, each tuned for its role.
All model names are configurable in `config.yaml` with no code changes.

| Role | Config key | Default | Used for |
|---|---|---|---|
| Main SQL | `llm.model` | `gemini-2.5-flash` | SQL generation |
| Report | `llm.model` + `report_max_output_tokens` | `gemini-2.5-flash` / 1024 tokens | Report writing (token-capped) |
| Lightweight | `llm.classification_model` | `gemini-2.5-flash-lite` | Intent classification, SQL correction, context summarisation |
| Embedding | `llm.embedding_model` | `models/text-embedding-004` | Golden Bucket similarity search |

> **Note:** `gemini-2.0-flash` and `gemini-2.0-flash-lite` are deprecated as of March 2026
> and no longer available to new users. Use `gemini-2.5-flash-lite` as the lightweight model.

---

## Architecture

```
User input (CLI)
      │
      ▼
┌──────────────────────────────────────────────┐
│ classify_intent  [gemini-2.5-flash-lite]      │
│ · ANALYSIS / DESTRUCTIVE / OUT_OF_SCOPE       │
│ · extracts client name for DESTRUCTIVE ops    │
│ · rejects DB/DDL ops before confirmation      │
└────────┬─────────────────────────────────────┘
         │
   ┌─────▼───────────────────────────────────┐
   │  _route_intent                           │
   │  ├─ error / quota    ──► mask_and_format │
   │  ├─ out_of_scope     ──► mask_and_format │
   │  ├─ destructive      ──► confirmation_gate│
   │  └─ analysis         ──► execute_analysis│
   └──────────────────────────────────────────┘
         │                        │
         │                ┌───────▼────────┐
         │                │ confirmation_  │
         │                │ gate (CLI)     │
         │                └───────┬────────┘
         │                        │ confirmed
         │                ┌───────▼────────┐
         │                │ execute_       │
         │                │ destructive    │
         │                └───────┬────────┘
         │                        │
  ┌──────▼──────────────────────┐ │
  │  execute_analysis           │ │
  │  1. Golden Bucket search    │ │  (TTL-cached)
  │  2. Build context           │ │  (verbatim + summary)
  │  3. SQL generation          │ │  [gemini-2.5-flash]
  │  4. SQL validation          │ │
  │  5. BigQuery execution      │ │  (PII strip at data layer)
  │     └─ self-correction      │ │  [gemini-2.5-flash-lite]
  │  6. Report generation       │ │  [gemini-2.5-flash, ≤1024 tokens]
  │  7. Log candidate trio      │ │
  └──────┬──────────────────────┘ │
         └────────────┬───────────┘
                      ▼
             ┌────────────────┐
             │ mask_and_format │  PII regex pass + user format prefs
             └────────┬───────┘
                      ▼
                 CLI output
```

### Golden Bucket governance
The Golden Bucket is **expert-curated**, not auto-populated from user queries.

1. Successful queries are logged to `data/candidate_trios.jsonl` with `"promoted": false`
2. A human expert reviews and sets `"promoted": true` on approved entries
3. Run `python scripts/promote_trios.py` to ingest approved entries
4. Promoted entries are marked `"ingested": true` — re-runs are idempotent
5. Cache is invalidated automatically when new trios are added

---

## Setup

### Prerequisites
- Python 3.11+
- Google Cloud account with BigQuery access
- Gemini API key (free from https://aistudio.google.com)

### Install
```bash
git clone https://github.com/codebeetl/ai-test.git
cd ai-test
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in your keys
```

### GCP Auth
```bash
gcloud auth application-default login
```

### Run
```bash
python main.py
```

On startup the agent will:
1. **Quota check** — probes the Gemini API; exits cleanly with reset time if daily quota is exhausted
2. Load Golden Bucket startup hints
3. Display the interactive banner

---

## Configuration

All tuneable parameters live in `config.yaml`. No code changes are needed to adjust models,
limits, safety phrases, or caching behaviour.

```yaml
llm:
  model: "gemini-2.5-flash"            # SQL generation + report writing
  classification_model: "gemini-2.5-flash-lite"  # classify, correct, summarise
  correction_model: "gemini-2.5-flash-lite"       # SQL self-correction rewrites
  embedding_model: "models/text-embedding-004"    # Golden Bucket embeddings
  temperature: 0.2
  report_max_output_tokens: 1024       # caps report verbosity and output tokens

agent:
  report_max_rows: 20                  # rows sent to report LLM (token saving)
  context_verbatim_turns: 2            # recent turns kept verbatim in SQL prompt
  context_summary_enabled: true        # summarise older turns via flash-lite
  golden_bucket_cache_ttl_s: 300       # cache TTL in seconds (0 = disabled)
  golden_bucket_cache_size: 50         # max cached entries

safety:
  confirm_phrase: "YES DELETE"         # exact phrase required for destructive ops
  default_user_id: "manager_a"
  pii_columns: [email, phone, ...]     # columns stripped at data layer

resilience:
  llm_max_attempts: 5
  bq_max_attempts: 3
  sql_max_retries: 3                   # SQL self-correction attempts
```

`src/config/persona.yaml` controls agent tone and style independently of the main config.

---

## Destructive Operations

The agent supports deletion of **saved reports** only — not database tables or raw data.

Valid: `"Delete saved reports for Acme Corp"`
Invalid: `"Delete the users table"` → rejected with a clear message before confirmation

The flow:
1. `classify_intent` extracts the client name and validates the request
2. If no client name is found, the request is rejected immediately (no confirmation prompt shown)
3. If valid, operator must type the exact `confirm_phrase` from `config.yaml`
4. `execute_destructive` uses the pre-extracted client name — no additional LLM call

---

## CLI Commands

| Command | Effect |
|---|---|
| Any question | Natural language → SQL → report query |
| `/format table` | Switch output to markdown table |
| `/format bullets` | Switch output to bullet list |
| `/whoami <user_id>` | Switch active user (loads their saved preferences) |
| `/quit` or `/exit` | Exit and write metrics snapshot |

---

## Tests

```bash
pytest tests/test_qa_evals.py
```

Tests hit the real BigQuery public dataset (read-only, free tier).
PII masking is tested at the data layer — `run_bigquery_query` strips PII columns
before returning results, so tests verify masking regardless of which code path is used.

Requires GCP credentials — set `GOOGLE_APPLICATION_CREDENTIALS` or use ADC.

---

## CI

| Job | Tool | Trigger |
|---|---|---|
| Lint | `ruff check` (line-length 100, py311) | Push / PR to main |
| Canary smoke | `pytest tests/test_qa_evals.py` | Push / PR to main |

Canary smoke requires `GCP_SA_KEY` and `GEMINI_API_KEY` as GitHub repository secrets.
See `.github/workflows/ci.yml` for the full workflow definition.

---

## Monitoring

All agent activity is written as structured JSON to `data/agent.log`.
The CLI console is **fully silent** — internal warnings, retries, and errors go to the log file
only. User-facing errors are delivered via the agent response, not log output.

```bash
tail -f data/agent.log | python -m json.tool
```

Log files rotate at 5 MB (3 backups kept). A metrics snapshot is written to
`data/metrics_snapshot.json` on exit with session counters and average latency.

---

## File Structure

```
├── config.yaml                    # all key variables — edit here
├── main.py                        # CLI entry point
├── scripts/
│   └── promote_trios.py           # expert Golden Bucket promotion tool
├── src/
│   ├── agent/
│   │   ├── graph.py               # LangGraph state machine + routing
│   │   ├── nodes.py               # node implementations (3 LLM instances)
│   │   └── state.py               # AgentState TypedDict
│   ├── config/
│   │   ├── settings.py            # Pydantic settings (LLM, agent, safety, resilience)
│   │   └── persona.yaml           # agent tone and style (edit without redeployment)
│   ├── memory/
│   │   ├── golden_bucket.py       # expert trio store (SQLite + FAISS)
│   │   ├── user_prefs.py          # per-user format preferences
│   │   └── reports_store.py       # saved report persistence
│   ├── observability/
│   │   ├── logger.py              # rotating JSON log; console threshold = CRITICAL
│   │   ├── metrics.py             # session counters + latency tracking
│   │   └── progress.py            # single-line CLI progress indicators
│   ├── oversight/
│   │   └── confirmation_flow.py   # destructive op CLI confirmation
│   ├── resilience/
│   │   ├── quota_check.py         # startup API quota probe
│   │   ├── quota_guard.py         # daily vs rate-limit classifier; reset time calc
│   │   ├── retry.py               # tenacity back-off decorator
│   │   └── sql_self_correct.py    # SQL rewrite loop with schema context
│   ├── safety/
│   │   ├── intent_guard.py        # pre-LLM keyword/regex filter
│   │   └── pii_masker.py          # two-pass PII removal (regex + column drop)
│   └── tools/
│       ├── golden_bucket_tool.py  # similarity search + TTL cache
│       ├── query_tool.py          # BigQuery execution + PII strip at data layer
│       ├── reports_tool.py        # saved reports CRUD
│       └── schema_tool.py         # BigQuery schema introspection
└── tests/
    └── test_qa_evals.py           # golden query + PII leak smoke tests
```
