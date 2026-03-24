# Observability Guide

## Log Fields

Every log line emitted by the agent is JSON-structured with these base fields:

| Field | Description |
|---|---|
| `timestamp` | ISO-8601 UTC timestamp |
| `level` | DEBUG / INFO / WARNING / ERROR |
| `logger` | Python module name (e.g. `src.tools.query_tool`) |
| `message` | Human-readable summary |

## Key Events and Metrics

| Event | Level | Extra Fields |
|---|---|---|
| BigQuery query executed | INFO | `sql_preview` |
| SQL retry | WARNING | `attempt`, `error` |
| SQL failed (max retries) | ERROR | `error` |
| PII masked in output | WARNING | — |
| PII columns dropped | WARNING | `cols` |
| Destructive op confirmed | WARNING | `action` |
| Destructive op aborted | INFO | — |
| Trio saved to Golden Bucket | INFO | `question_preview` |
| BigQuery init failed | ERROR | `error` |

## Sample Log Lines

```json
{"timestamp":"2026-03-24T16:00:01Z","level":"INFO","logger":"src.tools.query_tool","message":"run_bigquery_query invoked","sql_preview":"SELECT COUNT(*) AS total_orders FROM"}
{"timestamp":"2026-03-24T16:00:02Z","level":"WARNING","logger":"src.resilience.sql_self_correct","message":"SQL attempt failed","attempt":1,"error":"Syntax error: Expected end of input but got keyword FROM"}
{"timestamp":"2026-03-24T16:00:03Z","level":"WARNING","logger":"src.safety.pii_masker","message":"PII detected and masked in output string"}
{"timestamp":"2026-03-24T16:00:04Z","level":"WARNING","logger":"src.oversight.confirmation_flow","message":"Destructive operation confirmed","action":"Delete 3 reports for Client X"}
```

## Extending Observability

To send metrics to Datadog or GCP Monitoring, replace `JsonFormatter`
in `src/observability/logger.py` with a handler that ships to your
preferred log sink. The JSON structure is pre-designed to be ingested
without additional parsing.
