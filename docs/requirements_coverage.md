# Requirements Coverage

Maps each assignment requirement to concrete modules.

## 1. Hybrid Intelligence
- `src/memory/golden_bucket.py` — FAISS similarity search over past trios
- `src/tools/golden_bucket_tool.py` — `@tool` search_golden_bucket, save_trio
- `src/agent/nodes.py::execute_analysis` — injects similar trios into SQL prompt

## 2. PII Masking
- `src/safety/pii_masker.py` — regex text masking + column-level DataFrame drop
- `src/agent/nodes.py::mask_and_format` — single choke point, always applied

## 3. High-Stakes Oversight
- `src/oversight/confirmation_flow.py` — require_confirmation() with YES DELETE phrase
- `src/tools/reports_tool.py::delete_reports_by_client` — calls confirmation before delete

## 4. Continuous Improvement
- `src/memory/user_prefs.py` — per-user table/bullets preference
- `src/tools/golden_bucket_tool.py::save_trio` — grows Golden Bucket after success

## 5. Resilience
- `src/resilience/sql_self_correct.py` — with_sql_self_correction(), MAX_SQL_RETRIES=2
- `src/resilience/retry.py` — generic tenacity back-off decorator

## 6. Quality Assurance
- `tests/unit/` — 6 focused unit tests
- `tests/integration/` — graph compilation and routing
- `tests/canary/` — end-to-end PII-free output under 30s

## 7. Observability
- `src/observability/logger.py` — JSON structured logging
- Key events: PII masked, SQL retries, destructive op confirmed/aborted

## 8. Agility / Persona Management
- `src/config/persona.yaml` — non-developer editable tone/style hints
- `src/config/settings.py` — loaded at runtime, no redeploy needed
