# Retail Data Agent

> ⚠️ **AI Assistance Disclosure**: This prototype was scaffolded with the
> assistance of AI tooling (Perplexity + GitHub Copilot). All architecture
> decisions, code review, and testing were performed by the author.

An internal AI data analysis agent for non-technical retail executives,
built on LangGraph + BigQuery. Supports natural language queries over
`bigquery-public-data.thelook_ecommerce`.

## Features

- Natural language → SQL via Gemini
- PII masking (emails, phones) enforced on all outputs
- High-stakes confirmation flow for destructive operations (GDPR)
- SQL self-correction with capped retries
- Golden Bucket hybrid intelligence (FAISS + SQLite)
- Per-user output format preferences

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
cp .env.example .env   # Fill in your keys
```

### GCP Auth
```bash
gcloud auth application-default login
```

### Run
```bash
python main.py
```

## Tests
```bash
pytest tests/unit
pytest tests/integration
```

## Configuration
- **LLM/persona**: edit `config.yaml` and `src/config/persona.yaml` — no redeploy needed
- **Environment**: see `.env.example`
