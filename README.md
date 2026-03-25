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
| Natural language → SQL | Two-step Gemini pipeline: SQL-only prompt then separate report prompt |
| PII masking | Regex + column-level removal on all outputs; column list in `config.yaml` |
| Destructive op confirmation | Operator must type exact phrase (configurable in `config.yaml`) |
| SQL self-correction | LLM rewrites failed SQL up to `sql_max_retries` times (configurable) |
| Golden Bucket | Expert-curated Q→SQL→Report trios for hybrid retrieval (SQLite + embeddings) |
| Per-user output format | `table` or `bullets` preference stored per user |
| Quota guard | Distinguishes daily quota exhaustion from TPM/RPM soft limits; shows reset time |
| Startup quota check | Probes the API before the main loop; exits cleanly if daily quota is exhausted |
| Progress feedback | Single-line overwriting stage indicators during query execution |
| Metrics snapshot | Session counters and latencies written to `data/metrics_snapshot.json` on exit |
| Structured logging | All activity written as JSON to `data/agent.log`; noisy third-party loggers suppressed |

---

## Architecture

```
User input (CLI)
      │
      ▼
┌─────────────────┐
│ classify_intent  │  Gemini: ANALYSIS / DESTRUCTIVE / OUT_OF_SCOPE
└────────┬────────┘
         │
   ┌─────┴──────────────────────┐
   │ error/quota?  out_of_scope? │
   └──┬─────────────────────────┘
      │
  ┌───┴──────────┐   ┌──────────────────────┐
  │ execute_     │   │ confirmation_gate     │
  │ analysis     │   │ (destructive ops)     │
  └──────┬───────┘   └──────────┬────────────┘
         │                      │ confirmed?
         │             ┌────────┴────────┐
         │             │ execute_        │
         │             │ destructive     │
         │             └────────┬────────┘
         └──────────────────────┘
                       │
               ┌───────▼────────┐
               │ mask_and_format │  PII masking + user format prefs
               └───────┬────────┘
                       │
                  CLI output
```

### execute_analysis flow
1. **Golden Bucket search** — retrieve up to 3 similar expert Q→SQL trios
2. **SQL generation** — dedicated SQL-only prompt (no persona, no prose)
3. **SQL validation** — rejects response if it does not start with `SELECT`/`WITH`
4. **BigQuery execution** — with self-correction retry loop
5. **Report generation** — separate LLM call applies persona to the result data
6. **Candidate trio logging** — successful queries appended to `data/candidate_trios.jsonl` for expert review

### Golden Bucket governance
The Golden Bucket is **expert-curated**, not auto-populated from user queries.

1. Successful queries are logged to `data/candidate_trios.jsonl` with `"promoted": false`
2. A human expert reviews the file and sets `"promoted": true` on approved entries
3. Run `python scripts/promote_trios.py` to ingest approved entries into the bucket
4. Promoted entries are marked `"ingested": true` so re-runs are idempotent

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
1. Check API quota — exits with a clear message and reset time if daily quota is exhausted
2. Load Golden Bucket startup hints
3. Display the interactive banner

---

## Configuration

**`config.yaml`** is the single source of truth for all key variables:

```yaml
llm:
  model: "gemini-2.5-flash"          # main LLM
  correction_model: "gemini-2.5-flash" # SQL self-correction model
  embedding_model: "models/embedding-001" # Golden Bucket embeddings
  temperature: 0.2

safety:
  confirm_phrase: "YES DELETE"        # destructive op confirmation phrase
  default_user_id: "manager_a"        # default CLI user
  pii_columns: [email, phone, ...]    # columns always stripped from output

resilience:
  llm_max_attempts: 5                 # LLM retry attempts
  bq_max_attempts: 3                  # BigQuery retry attempts
  sql_max_retries: 2                  # SQL self-correction rewrites
```

**`src/config/persona.yaml`** controls the agent's communication style — editable
without redeployment:

```yaml
tone: "professional and concise"
style_hints:
  - "Use bullet points for lists of more than 3 items"
  - "Always include a one-sentence executive summary"
```

---

## CLI Commands

| Command | Effect |
|---|---|
| Any question | Runs a natural language → SQL → report query |
| `/format table` | Switch output to markdown table |
| `/format bullets` | Switch output to bullet list |
| `/whoami <user_id>` | Switch active user (loads their saved preferences) |
| `/quit` or `/exit` | Exit and write metrics snapshot |

---

## Tests

```bash
pytest tests/test_qa_evals.py        # BigQuery golden query + PII leak tests
```

Tests hit the real BigQuery public dataset (read-only, free tier).
Ensure GCP credentials are configured before running.

---

## Monitoring

All agent activity is written as structured JSON to `data/agent.log`.
The CLI output is kept clean — noisy third-party loggers (tenacity, langchain,
google-api-core) are suppressed from the console and redirected to the log file.

```bash
tail -f data/agent.log | python -m json.tool
```

Log files rotate automatically at 5 MB (3 backups kept):

```
data/agent.log      ← current
data/agent.log.1    ← previous
data/agent.log.2    ← older
```

A session metrics snapshot is written to `data/metrics_snapshot.json` on exit,
containing counters (queries run, errors, quota events) and average latency.

---

## File Structure

```
├── config.yaml                    # all key variables — edit here
├── main.py                        # CLI entry point
├── scripts/
│   └── promote_trios.py           # expert Golden Bucket promotion tool
├── src/
│   ├── agent/
│   │   ├── graph.py               # LangGraph state machine
│   │   ├── nodes.py               # node implementations
│   │   └── state.py               # AgentState TypedDict
│   ├── config/
│   │   ├── settings.py            # Pydantic settings loader
│   │   └── persona.yaml           # agent tone and style
│   ├── memory/
│   │   ├── golden_bucket.py       # expert trio store (SQLite + embeddings)
│   │   ├── user_prefs.py          # per-user format preferences
│   │   └── reports_store.py       # saved report persistence
│   ├── observability/
│   │   ├── logger.py              # structured JSON logging setup
│   │   ├── metrics.py             # session counters + latency tracking
│   │   └── progress.py            # single-line CLI progress indicators
│   ├── oversight/
│   │   └── confirmation_flow.py   # destructive op confirmation prompt
│   ├── resilience/
│   │   ├── quota_check.py         # startup API quota probe
│   │   ├── quota_guard.py         # quota/rate-limit error classifier
│   │   ├── retry.py               # tenacity back-off decorator
│   │   └── sql_self_correct.py    # SQL rewrite retry loop
│   ├── safety/
│   │   ├── intent_guard.py        # out-of-scope intent filtering
│   │   └── pii_masker.py          # regex + column-level PII removal
│   └── tools/
│       ├── golden_bucket_tool.py  # LangChain Golden Bucket search tool
│       ├── query_tool.py          # BigQuery execution tool
│       ├── reports_tool.py        # saved reports CRUD tool
│       └── schema_tool.py         # BigQuery schema introspection tool
└── tests/
    └── test_qa_evals.py           # golden query smoke tests
```
