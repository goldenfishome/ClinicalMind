# ClinicalMind — Clinical Trial RAG Agent
## Project Requirements Document

**Version:** 1.2  
**Last Updated:** 2026-05-22  
**Author:** Yujia (Sian) Jin  
**Status:** Phase 1–2 complete; Phase 3 in progress

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Goals & Non-Goals](#2-goals--non-goals)
3. [Data Sources](#3-data-sources)
4. [Functional Requirements](#4-functional-requirements)
5. [Non-Functional Requirements](#5-non-functional-requirements)
6. [System Architecture](#6-system-architecture)
7. [Tech Stack](#7-tech-stack)
8. [Agent Design](#8-agent-design)
9. [RAG Design](#9-rag-design)
10. [Project Structure](#10-project-structure)
11. [Phased Delivery Plan](#11-phased-delivery-plan)
12. [Out of Scope](#12-out-of-scope)
13. [Open Questions](#13-open-questions)

---

## 1. Project Overview

**ClinicalMind** is an AI-powered clinical trial intelligence assistant that allows users to query and explore public clinical trial data through natural language. The system combines a Retrieval-Augmented Generation (RAG) pipeline with an agentic layer that dynamically decides how to answer each question — using semantic search, structured database queries, or live web search depending on what the question requires.

### Problem Statement

Clinical trial information is vast, scattered, and difficult to navigate. Researchers, analysts, and healthcare professionals spend significant time manually searching ClinicalTrials.gov for eligibility criteria, trial designs, outcomes, and protocol details. This project automates that process through a conversational AI interface backed by a structured, production-aware data pipeline.

### Target Audience

- Clinical researchers and analysts who need fast access to trial protocols and eligibility criteria (primary)
- Data scientists and ML engineers exploring clinical trial data
- Portfolio reviewers and technical interviewers

---

## 2. Goals & Non-Goals

### Goals

- Build an end-to-end RAG pipeline ingesting real public clinical trial data
- Implement a LangGraph agent with multiple tools that routes questions intelligently
- Demonstrate production-aware engineering: modular code, structured logging, error handling
- Deliver a live-demo-able Streamlit UI
- Produce a clean, well-documented GitHub repository suitable for a portfolio

### Non-Goals (explicitly deferred)

- MLflow experiment tracking and model registry (Phase 2)
- Docker containerization and CI/CD (Phase 2)
- User authentication and access control (Phase 2)
- ML outcome prediction model (Phase 2)
- Full RAGAS evaluation suite (Phase 2 — Phase 1 includes a minimal 20-question eval)
- Real-time data streaming
- Multi-user concurrency handling
- Fine-tuning any LLM

---

## 3. Data Sources

### Primary Data Source: ClinicalTrials.gov API (v2)

| Property | Detail |
|---|---|
| URL | https://clinicaltrials.gov/api/v2/studies |
| Format | JSON |
| Authentication | None required (public API) |
| Volume | 500,000+ registered trials |
| Cost | Free |
| Key fields | NCT ID, title, status, phase, conditions, interventions, eligibility criteria, sponsor, start/completion dates, locations, outcomes |

**API example call:**
```
GET https://clinicaltrials.gov/api/v2/studies?query.cond=diabetes&pageSize=100
```

### Secondary Data Source: AACT Database (optional enrichment)

| Property | Detail |
|---|---|
| URL | https://aact.ctti-clinicaltrials.org |
| Format | PostgreSQL database snapshot (downloadable) |
| Authentication | Free registration required |
| Volume | Full relational dump of ClinicalTrials.gov |
| Cost | Free |
| Use case | Structured SQL queries on trial metadata |

> **Decision:** Start with ClinicalTrials.gov API only. Add AACT if SQL tool is implemented in Phase 1.

### Initial Data Scope

To keep Phase 1 manageable, ingest trials filtered to:
- **Conditions:** Diabetes, Cardiac Arrest, Hypertension (3 domains for variety)
- **Status:** Completed, Recruiting, Active not recruiting
- **Volume target:** 5,000–10,000 trials for initial build  
- **Actual ingested:** 26,020 unique trials, 115,320 chunks

---

## 4. Functional Requirements

### FR-01: Data Ingestion
- The system shall fetch clinical trial data from the ClinicalTrials.gov v2 API
- The system shall support pagination to retrieve large result sets
- The system shall extract and store key fields: NCT ID, title, phase, conditions, interventions, eligibility text, sponsor, status, dates
- The system shall handle API errors gracefully with retry logic (max 3 retries, exponential backoff)
- The system shall save raw JSON responses locally before processing (raw storage layer)
- The system shall be re-runnable without creating duplicates (idempotent ingestion)

### FR-02: Text Processing & Chunking
- The system shall extract free-text fields (eligibility criteria, brief summary, detailed description) for embedding
- The system shall chunk long text fields using section-based chunking (not fixed-length)
- Each chunk shall retain metadata: NCT ID, source field, section heading, trial title, phase, condition
- The system shall generate embeddings using OpenAI text-embedding-3-small model

### FR-03: Vector Storage
- The system shall store embeddings in PostgreSQL with the pgvector extension
- The system shall store chunk text, embeddings, and metadata in the same table
- The system shall support cosine similarity search for retrieval
- The system shall support metadata filtering (e.g. filter by condition, phase, status)

### FR-04: RAG Question Answering
- The system shall retrieve the top-k most relevant chunks for a given user query (default k=5)
- The system shall construct a prompt grounding GPT-4o strictly on retrieved context
- The system shall instruct the model to cite the NCT ID of source trials in its answer
- The system shall return a structured response: answer text + list of source trials used

### FR-05: Agent Routing
- The system shall implement a LangGraph agent that receives a user question and decides which tool(s) to invoke
- The agent shall support 3 tools: RAG Search Tool, Web Search Tool, and SQL Lookup Tool
- The agent shall pass tool results to GPT-4o for synthesis into a final answer
- The agent shall handle tool failures gracefully without crashing

### FR-06: User Interface
- The system shall provide a Streamlit web interface
- The UI shall have a chat-style input and response area
- The UI shall display which tool(s) the agent used for each answer (transparency)
- The UI shall display source trial NCT IDs cited in the answer
- The UI shall include example questions to guide new users

### FR-07: Logging
- The system shall log all ingestion runs with record counts, errors, and timestamps
- The system shall log all agent invocations with the question, tools used, and response time
- Logs shall be written to both console and a local log file

---

## 5. Non-Functional Requirements

### NFR-01: Code Quality
- All Python code shall follow PEP8 standards
- All modules shall have docstrings
- No hardcoded credentials — all secrets via environment variables (.env file)
- Code shall be modular: ingestion, embedding, retrieval, agent, UI in separate modules

### NFR-02: Reproducibility
- A requirements.txt or pyproject.toml shall pin all dependency versions
- A .env.example file shall document all required environment variables
- The README shall include step-by-step setup instructions from a clean environment

### NFR-03: Error Handling
- API failures shall be caught, logged, and retried before raising
- LLM API failures shall return a graceful error message to the user, not a crash
- Missing or malformed trial fields shall be handled with defaults, not exceptions

### NFR-04: Performance (prototype targets)
- Ingestion of 5,000 trials shall complete within 30 minutes
- A single RAG query shall return a response within 10 seconds
- Vector search over 50,000 chunks shall complete within 2 seconds

### NFR-05: Portfolio Presentation
- The GitHub repository shall have a clear README with architecture diagram
- The README shall explain key design decisions and trade-offs
- The README shall include a live demo URL (Streamlit Community Cloud deployment)
- The README shall include LangSmith trace screenshots demonstrating agent routing behavior
- A short demo video (2-3 minutes) shall be recorded showing the system in action

---

## 6. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DATA LAYER                           │
│                                                             │
│  ClinicalTrials.gov API                                     │
│          │                                                  │
│          ▼                                                  │
│  Raw JSON Storage (local /data/raw/)                        │
│          │                                                  │
│          ▼                                                  │
│  Text Extraction + Chunking (Python)                        │
│          │                                                  │
│          ▼                                                  │
│  Embedding Generation (OpenAI text-embedding-3-small)       │
│          │                                                  │
│          ▼                                                  │
│  PostgreSQL + pgvector (chunks + embeddings + metadata)     │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                       AGENT LAYER                           │
│                                                             │
│  User Question                                              │
│          │                                                  │
│          ▼                                                  │
│  LangGraph Agent (Orchestrator)                             │
│          │                                                  │
│    ┌─────┴──────────────────┐                               │
│    ▼                        ▼                    ▼          │
│  RAG Search Tool       SQL Lookup Tool     Web Search Tool  │
│  (pgvector similarity) (structured meta)  (Tavily API)      │
│    │                        │                    │          │
│    └─────────────┬──────────┘────────────────────┘          │
│                  ▼                                          │
│          GPT-4o Synthesis                                   │
│                  │                                          │
│                  ▼                                          │
│     Grounded Answer + Source Citations                      │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                       │
│                                                             │
│             Streamlit Chat UI                               │
│    (answer + tool used + cited NCT IDs)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Tech Stack

| Layer | Tool | Version | Justification |
|---|---|---|---|
| Language | Python | 3.11+ | Industry standard for AI/ML |
| LLM | GPT-4o (OpenAI API) | latest | Best grounded answering; industry standard |
| Embeddings | text-embedding-3-small | latest | Cost-efficient; strong performance |
| Vector DB | PostgreSQL + pgvector | pg16, pgvector 0.7+ | Combines structured + semantic in one system |
| Agent framework | LangGraph | 0.2+ | Most in-demand agent framework; stateful control flow |
| LLM orchestration | LangChain | 0.3+ | Standard retrieval + prompt orchestration |
| Web search | Tavily API | latest | Designed for LLM agent use; clean results |
| Observability | LangSmith | latest | Agent trace visualization; demonstrates production awareness |
| RAG evaluation | RAGAS | latest | Measures retrieval recall and answer faithfulness |
| UI | Streamlit | 1.35+ | Fast, demo-friendly, widely recognized |
| UI hosting | Streamlit Community Cloud | latest | Free live deployment; shareable demo URL |
| Data processing | Pandas | 2.0+ | Standard tabular data handling |
| HTTP client | httpx | 0.27+ | Async-capable; better than requests for retries |
| Logging | Python logging (structlog) | latest | Structured JSON logs |
| Config | python-dotenv | latest | Environment variable management |
| Testing | pytest | 8.0+ | Standard Python testing |
| DB driver | psycopg2 / asyncpg | latest | PostgreSQL Python adapter |

---

## 8. Agent Design

### Agent Type
**LangGraph ReAct Agent** — uses the Reason + Act loop to decide which tools to call based on the user question.

### Agent State
```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # full conversation history
    question: str           # original user question (never mutated)
    sources:  list[str]     # NCT IDs cited across all tool calls this run
```

### Tools

#### Tool 1: RAG Search Tool
- **Trigger:** Questions about trial protocols, eligibility criteria, interventions, study design
- **Input:** Natural language query string
- **Process:** Embed query → cosine similarity search in pgvector → return top-5 chunks
- **Output:** List of relevant text chunks with NCT ID metadata
- **Example question:** "What are common exclusion criteria for Phase 3 diabetes trials?"

#### Tool 2: Web Search Tool
- **Trigger:** Questions requiring current information beyond the ingested dataset
- **Input:** Search query string
- **Process:** Call Tavily API → return top results
- **Output:** List of web snippets with URLs
- **Example question:** "What are the latest FDA guidelines on diabetes trial design in 2025?"

#### Tool 3: SQL Lookup Tool
- **Trigger:** Questions requiring counts, aggregations, or filtered lists
- **Input:** Natural language question → LLM generates SQL → execute against PostgreSQL
- **Output:** Structured query results
- **Example question:** "How many Phase 2 cardiac arrest trials are currently recruiting?"

### Agent Decision Logic
```
If question requires semantic understanding of trial text → RAG Search Tool
If question requires current/live information             → Web Search Tool
If question requires counts or structured filtering       → SQL Lookup Tool
If question requires combining multiple sources           → Multiple tools in sequence
```

---

## 9. RAG Design

### Chunking Strategy
- **Method:** Section-based chunking (split on field boundaries, not character count)
- **Fields chunked separately:** brief_summary, eligibility_criteria, detailed_description
- **Rationale:** Preserves semantic meaning per section; improves retrieval precision vs. fixed-length chunking
- **Chunk size target:** 300–800 tokens per chunk

### Metadata Per Chunk
```python
{
  "nct_id": "NCT12345678",
  "title": "Trial title",
  "phase": "Phase 3",
  "condition": "Diabetes",
  "status": "Recruiting",
  "source_field": "eligibility_criteria",
  "section": "Inclusion Criteria",
  "sponsor": "NIH"
}
```

### Retrieval
- **Similarity metric:** Cosine similarity
- **Top-k:** 5 chunks (default; configurable)
- **Metadata filtering:** Support filter by condition, phase, status before similarity search

### Prompt Design
```
System: You are a clinical trial research assistant. Answer the user's question 
based ONLY on the provided context. If the answer is not in the context, say so. 
Always cite the NCT ID of trials you reference.

Context:
[retrieved chunks with NCT IDs]

User question: [question]
```

### Embedding Model Choice
- **Model:** text-embedding-3-small (OpenAI)
- **Dimensions:** 1536
- **Rationale:** Lower cost than text-embedding-3-large; sufficient quality for this use case

---

## 10. Project Structure

```
clinicalmind/
│
├── README.md                    # Project overview, architecture, setup guide
├── REQUIREMENTS.md              # This document
├── CLAUDE.md                    # Claude Code guidance (local only, git-excluded)
├── requirements.txt             # Dependency version ranges
├── .env.example                 # Template for required environment variables
├── .gitignore                   # Exclude .env, data/, __pycache__, logs/*.log
│
├── data/
│   ├── raw/                     # Raw JSON pages from ClinicalTrials.gov API
│   └── processed/               # embedded_chunks.json cache (avoids re-embedding)
│
├── ingestion/
│   ├── __init__.py
│   ├── fetch_trials.py          # ClinicalTrials.gov API client + pagination + retry
│   ├── parse_trials.py          # Extract and clean fields from raw JSON
│   └── chunk_text.py            # Section-based chunking logic
│
├── embeddings/
│   ├── __init__.py
│   ├── embed.py                 # Generate embeddings via OpenAI API (batched)
│   └── store.py                 # Upsert chunks + embeddings into pgvector
│
├── retrieval/
│   ├── __init__.py
│   └── search.py                # Cosine similarity search with metadata filtering
│
├── agent/
│   ├── __init__.py
│   ├── logging_config.py        # Centralised structlog setup (console + file)
│   ├── rag.py                   # Core RAG function: retrieve → prompt → GPT-4o
│   ├── tools.py                 # rag_search, web_search, sql_lookup tool definitions
│   ├── graph.py                 # LangGraph ReAct agent + run_agent() entry point
│   └── prompts.py               # System prompts and prompt templates
│
├── database/
│   ├── __init__.py
│   ├── schema.sql               # pgvector table + IVFFlat index definitions
│   └── connection.py            # ThreadedConnectionPool + get_conn() context manager
│
├── ui/
│   └── app.py                   # Streamlit chat interface
│
├── tests/
│   ├── test_ingestion.py        # fetch_trials + parse_trials unit tests
│   ├── test_chunking.py         # chunk_text unit tests (23 tests)
│   ├── test_embeddings.py       # embed.py + store.py unit tests
│   ├── test_store.py            # store.py upsert + idempotency tests
│   ├── test_retrieval.py        # search.py unit tests
│   ├── test_agent.py            # rag.py + prompts.py unit tests
│   ├── test_graph.py            # graph.py (LangGraph agent) unit tests
│   ├── test_tools.py            # tools.py (all 3 tools) unit tests
│   ├── eval_dataset.json        # 20 hand-labeled Q&A pairs for RAGAS
│   ├── ragas_eval.py            # RAGAS evaluation runner (not collected by pytest)
│   └── ragas_results.json       # Latest RAGAS scores (gitignored)
│
└── logs/
    ├── .gitkeep                 # Keeps directory tracked; log files are gitignored
    └── clinicalmind.log         # Rotating JSON log (10 MB × 7 backups, gitignored)
```

---

## 11. Phased Delivery Plan

### Phase 1 — Data Ingestion & RAG (Week 1-2) ✅ COMPLETE
- [x] Set up PostgreSQL + pgvector locally
- [x] Build ClinicalTrials.gov API client with pagination and retry logic
- [x] Implement section-based text chunking
- [x] Generate and store embeddings in pgvector (26,020 trials, 115,320 chunks)
- [x] Build basic RAG Q&A (retrieval + GPT-4o response) — `agent/rag.py`
- [x] Wire in LangSmith for agent trace logging
- [x] Build minimal RAGAS evaluation: 20 hand-labeled Q&A pairs
  - Faithfulness: 0.515 | Answer Relevancy: 0.688 | Context Precision: 0.404
- [x] Validate with test questions manually

**Milestone:** ✅ Can ask a question and get a grounded answer with NCT citations; RAGAS scores documented

### Phase 2 — Agent Layer (Week 3-4) ✅ COMPLETE
- [x] Define LangGraph agent state and graph — `agent/graph.py`
- [x] Implement RAG Search Tool — `agent/tools.py` (calls `agent/rag.py`)
- [x] Implement Web Search Tool using Tavily API
- [x] Implement SQL Lookup Tool (text-to-SQL over PostgreSQL trial metadata)
- [x] Connect agent to GPT-4o for synthesis
- [x] Test agent routing with diverse question types (122 passing unit tests)
- [x] Verify LangSmith traces capture routing decisions across all three tools

**Milestone:** ✅ Agent correctly routes questions to appropriate tool(s); routing decisions visible in LangSmith

### Phase 3 — UI & Polish (Week 4-5) 🔄 IN PROGRESS
- [x] Build Streamlit chat interface — `ui/app.py`
- [x] Display tool used + source citations in UI (collapsible tool call expander)
- [x] Add example questions panel (sidebar)
- [x] Add structured logging throughout — `agent/logging_config.py`
- [x] Write pytest tests — 122 passing across 8 test files
- [ ] Deploy to Streamlit Community Cloud; add live demo URL to README
- [ ] Add LangSmith trace screenshots to README showing agent routing examples
- [ ] Write README with architecture diagram and setup guide
- [ ] Record demo video

**Milestone:** Fully demo-able project with live URL, clean GitHub repo, and documented eval scores

---

## 12. Out of Scope

The following are explicitly deferred to a future Phase 2 and should NOT be built in Phase 1:

| Item | Reason deferred |
|---|---|
| MLflow experiment tracking | Adds complexity; more relevant when ML model is added |
| Docker + CI/CD | Infrastructure overhead; add after core system is stable |
| User authentication | Not needed for portfolio demo |
| ML trial outcome predictor | Significant additional scope; plan for Phase 2 |
| Full RAGAS evaluation suite | Minimal eval (20 Q&A pairs) is in Phase 1; full suite deferred |
| Multi-agent system (CrewAI) | Overkill for current scope |
| Fine-tuning | Not needed; GPT-4o with RAG is sufficient |

---

## 13. Open Questions — Resolved

| # | Question | Decision |
|---|---|---|
| 1 | Use OpenAI embeddings (paid) or sentence-transformers (free local)? | **OpenAI text-embedding-3-small** — best quality/cost ratio; consistent with GPT-4o stack |
| 2 | Use local PostgreSQL or a free cloud DB (e.g. Supabase free tier)? | **Local PostgreSQL** for development; Streamlit Community Cloud deployment will need a hosted DB |
| 3 | Tavily API for web search or SerpAPI? | **Tavily** — purpose-built for LLM agents, returns structured results + synthesised answer |
| 4 | Add AACT PostgreSQL dump for SQL tool, or generate SQL over the ingested pgvector table? | **Generate SQL over the ingested `trial_chunks` table** — avoids a second data store; same DB serves both vector search and structured queries |
| 5 | Streamlit Community Cloud for hosting demo, or local only? | **Streamlit Community Cloud** — free tier, shareable URL, no infrastructure management |

---

*Document maintained by Yujia (Sian) Jin. Update version number and date on each revision.*
