# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project

**ClinicalMind** is a clinical trial intelligence assistant. It ingests data from the ClinicalTrials.gov v2 API, stores embeddings in PostgreSQL+pgvector, and exposes a LangGraph ReAct agent with three tools (RAG search, SQL lookup, web search) through a Streamlit UI. The primary goal is a production-aware, demo-able portfolio project.

Full requirements are in [REQUIREMENTS.md](REQUIREMENTS.md).

---

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill in environment variables
cp .env.example .env

# Run the Streamlit UI
streamlit run ui/app.py

# Ingest clinical trial data (fetch → parse → chunk → embed → store)
python -m ingestion.fetch_trials
python -m ingestion.parse_trials
python -m ingestion.chunk_text
python -m embeddings.embed
python -m embeddings.store

# Smoke-test the agent directly (no UI)
python -m agent.graph

# Run RAGAS evaluation (all 20 questions, ~2 hours)
python -m tests.ragas_eval

# Run RAGAS quick smoke test (first 2 questions, ~15 min)
python -m tests.ragas_eval --n 2

# Run all tests
pytest --ignore=tests/ragas_eval.py

# Run a single test file
pytest tests/test_retrieval.py

# Run a single test by name
pytest tests/test_chunking.py::test_section_split
```

---

## Architecture

### Data flow

```
ClinicalTrials.gov API
  → ingestion/fetch_trials.py    (paginated fetch, raw JSON to data/raw/)
  → ingestion/parse_trials.py    (extract NCT ID, title, phase, conditions,
                                  eligibility text, sponsor, status, dates)
  → ingestion/chunk_text.py      (section-based chunking, not fixed-length;
                                  each chunk carries full metadata)
  → embeddings/embed.py          (OpenAI text-embedding-3-small, 1536-dim)
  → embeddings/store.py          (INSERT into pgvector table)
```

### Agent layer

`agent/graph.py` defines a LangGraph ReAct agent. The agent state (`AgentState` TypedDict) holds the full message history, original question, and cited NCT IDs. On each turn the agent picks one or more tools from `agent/tools.py`:

| Tool | When invoked | Backend |
|---|---|---|
| RAG Search | Semantic questions about trial text, eligibility, design | `agent/rag.py` → pgvector cosine similarity, top-5 chunks |
| SQL Lookup | Count/aggregation/filter questions | GPT-4o generates SQL → executes against PostgreSQL |
| Web Search | Questions requiring current information beyond the corpus | Tavily API |

After tool calls, GPT-4o synthesizes a final answer grounded on tool output, citing NCT IDs. System prompts live in `agent/prompts.py`.

`agent/rag.py` is the core RAG function used by the RAG Search Tool. It handles retrieval → prompt construction → GPT-4o call → NCT ID extraction. It is also independently runnable for smoke-testing without the agent.

`agent/logging_config.py` centralises all structlog configuration. Call `setup_logging()` once at app startup; every module then uses `log = structlog.get_logger()` only. Logs fan out to both console (coloured, human-readable) and `logs/clinicalmind.log` (rotating JSON, 10 MB × 7 backups).

### RAG design decisions

- **Section-based chunking** (not fixed-length): `brief_summary`, `eligibility_criteria`, and `detailed_description` are chunked on field boundaries to preserve semantic coherence. Target 300–800 tokens per chunk.
- **Metadata filtering before similarity search**: queries can be pre-filtered by condition, phase, or status before the cosine search runs.
- The RAG prompt instructs the model to answer *only* from retrieved context and always cite NCT IDs.

### Database

`database/schema.sql` defines the pgvector table. `database/connection.py` manages the PostgreSQL connection. The same database serves both the vector store (pgvector similarity) and the SQL Lookup Tool (structured metadata queries), so no second data store is needed.

### Observability & evaluation

- **LangSmith** is wired into the LangGraph agent for trace logging. Set `LANGCHAIN_API_KEY` and `LANGCHAIN_TRACING_V2=true` in `.env`.
- **RAGAS** evaluation runs against a 20-question hand-labeled dataset (`tests/eval_dataset.json`). Three metrics: Faithfulness (0.52), Answer Relevancy (0.69), Context Precision (0.40). Results saved to `tests/ragas_results.json`.

---

## Environment variables

See `.env.example`. Required keys:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | LLM and embedding calls |
| `DATABASE_URL` | PostgreSQL connection string |
| `TAVILY_API_KEY` | Web Search tool |
| `LANGCHAIN_API_KEY` | LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | Set to `true` to enable LangSmith |

---

## Key design constraints

- Ingestion must be **idempotent** — re-running must not create duplicate chunks or embeddings.
- API calls to ClinicalTrials.gov use **exponential backoff** with max 3 retries.
- LLM API failures must return a graceful error message to the user, not raise an unhandled exception.
- All secrets via environment variables — never hardcoded.
- Phase 1 data scope: Diabetes, Cardiac Arrest, Hypertension; Completed/Recruiting/Active statuses. Actual ingested: **26,020 unique trials, 115,320 chunks**.
