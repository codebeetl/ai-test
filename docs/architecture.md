# Retail Data Agent – Architecture

This document describes the high-level architecture of the retail data
analysis agent built on BigQuery and LangGraph.

## High-Level Diagram

```mermaid
flowchart TD
    subgraph CLI
        U[Store/Regional Manager]
        C[Python CLI main.py]
    end

    subgraph Agent[LangGraph Agent]
        CI[Classify Intent]
        EA[Execute Analysis]
        CD[Confirmation Gate]
        ED[Execute Destructive]
        MF[Mask and Format]
    end

    subgraph LLM[LLM and Embeddings]
        G[Gemini Chat]
        E[Gemini Embeddings]
    end

    subgraph Data[Data Sources]
        BQ[(BigQuery thelook_ecommerce)]
        GB[(Golden Bucket SQLite + FAISS)]
        SR[(Saved Reports SQLite)]
        UP[(User Prefs SQLite)]
    end

    subgraph Safety[Safety and Resilience]
        PM[PII Masker]
        SC[SQL Self-Correct]
        CF[Confirmation Flow]
    end

    U -->|Natural language| C
    C -->|invoke graph| CI
    CI -->|analysis| EA
    CI -->|destructive| CD
    EA -->|generate SQL| G
    EA -->|execute| BQ
    EA -->|similarity search| GB
    CD --> ED
    ED -->|delete| SR
    ED --> CF
    EA --> MF
    ED --> MF
    MF --> PM
    MF --> UP
    PM -->|final output| C
    SC -. wraps .- BQ
    G --> E
    GB --> E
```

## Components

### CLI (`main.py`)
Simple Opsfleet-style interactive loop. Maintains AgentState per session.

### Agent (LangGraph)
- **AgentState**: messages, user_id, pending_destructive_op, last_sql, retry_count, raw_result, final_output
- **Graph nodes**: classify_intent → execute_analysis / confirmation_gate → mask_and_format

### Core Requirements

| Requirement | Module |
|---|---|
| PII Masking | `src/safety/pii_masker.py` — column drop + regex |
| Resilience | `src/resilience/sql_self_correct.py` — LLM rewrite loop |
| High-Stakes Oversight | `src/oversight/confirmation_flow.py` — YES DELETE gate |
| Golden Bucket | `src/memory/golden_bucket.py` — FAISS + SQLite |
| User Preferences | `src/memory/user_prefs.py` — SQLite |
| Persona Management | `src/config/persona.yaml` — editable at runtime |
| Observability | `src/observability/logger.py` — JSON structured logs |
