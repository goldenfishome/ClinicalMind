"""
LangGraph tool implementations for the ClinicalMind agent.

Each function decorated with @tool becomes a callable the LLM can invoke.
The docstring is what GPT-4o reads to decide *when* to call the tool and
*what arguments* to pass — keep them precise and concrete.

Tool status:
  rag_search  ✅  Feature 13 — calls agent/rag.py ask()
  web_search  ✅  Feature 14 — Tavily web search
  sql_lookup  ✅  Feature 15 — text-to-SQL over PostgreSQL
"""

import os
import re

import structlog
from langchain_core.tools import tool
from openai import OpenAI

from agent.logging_config import setup_logging
from agent.rag import ask
from database.connection import get_conn

# Optional dependency — only needed when TAVILY_API_KEY is configured.
# Imported at module level so tests can patch agent.tools.TavilyClient.
try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None  # type: ignore[assignment,misc]

setup_logging()
log = structlog.get_logger()


# ---------------------------------------------------------------------------
# RAG Search Tool  (Feature 13)
# ---------------------------------------------------------------------------

@tool
def rag_search(
    query: str,
    condition: str | None = None,
    phase: str | None = None,
    status: str | None = None,
) -> str:
    """
    Search the clinical trial database for passages relevant to the query.

    Use this for ANY question about trial eligibility criteria, interventions,
    outcomes, trial design, phases, sponsors, or any free-text trial content.

    Optional filters narrow the search before semantic matching:
      condition : disease area, e.g. "Diabetes", "Hypertension", "Cardiac Arrest"
      phase     : trial phase, e.g. "PHASE1", "PHASE2", "PHASE3", "PHASE4"
      status    : enrollment status, e.g. "RECRUITING", "COMPLETED",
                  "ACTIVE_NOT_RECRUITING"

    Returns a cited answer synthesised from the top matching trial excerpts,
    with NCT IDs in parentheses after each claim.
    """
    result = ask(
        question  = query,
        top_k     = 5,
        condition = condition,
        phase     = phase,
        status    = status,
    )

    if result["error"]:
        return f"RAG search encountered an error: {result['error']}"

    if not result["chunks_used"]:
        return (
            "No relevant clinical trial data found for this query. "
            "Try rephrasing or removing filters."
        )

    sources_line = (
        "Trials cited: " + ", ".join(result["sources"])
        if result["sources"]
        else "No NCT IDs were cited in this answer."
    )
    return f"{result['answer']}\n\n{sources_line}"


# ---------------------------------------------------------------------------
# Web Search Tool  (Feature 14)
# ---------------------------------------------------------------------------

def _is_placeholder_key(key: str | None) -> bool:
    """Return True when the API key is missing or still set to the example value."""
    return not key or key.startswith("tvly-...")


@tool
def web_search(query: str) -> str:
    """
    Search the web for current information not available in the trial database.

    Use ONLY when the question clearly requires recent news, regulatory updates,
    drug approvals, or information that postdates the trial corpus.
    Do NOT use for questions answerable from trial text — use rag_search instead.
    """
    api_key = os.environ.get("TAVILY_API_KEY", "")

    if _is_placeholder_key(api_key):
        log.warning("web_search_skipped", reason="TAVILY_API_KEY not configured")
        return (
            "Web search is not available: TAVILY_API_KEY is not configured. "
            "Please add a real key to .env to enable this tool."
        )

    try:
        if TavilyClient is None:
            return "Web search is not available: tavily-python is not installed."
        client = TavilyClient(api_key=api_key)

        log.info("web_search_called", query=query[:80])
        response = client.search(
            query             = query,
            search_depth      = "basic",
            max_results       = 5,
            include_answer    = True,   # Tavily returns a short synthesised answer
        )

        # Format: synthesised answer first, then individual sources
        parts: list[str] = []

        if response.get("answer"):
            parts.append(response["answer"])

        results = response.get("results", [])
        if results:
            parts.append("\nSources:")
            for r in results:
                title   = r.get("title", "Untitled")
                url     = r.get("url", "")
                snippet = r.get("content", "")[:200].replace("\n", " ")
                parts.append(f"- {title}\n  {url}\n  {snippet}…")

        log.info("web_search_complete", n_results=len(results))
        return "\n".join(parts) if parts else "Web search returned no results."

    except Exception as exc:
        log.error("web_search_failed", error=str(exc))
        return f"Web search failed: {exc}"


# ---------------------------------------------------------------------------
# SQL Lookup Tool  (Feature 15)
# ---------------------------------------------------------------------------

# Schema description given to the LLM so it can write correct SQL.
# Deliberately concise — include only what's needed for valid queries.
_SQL_SCHEMA = """
TABLE: trial_chunks
One row = one text chunk from one clinical trial.
Use COUNT(DISTINCT nct_id) to count unique trials (many rows share the same nct_id).

COLUMNS:
  nct_id       TEXT    -- trial identifier, e.g. 'NCT04512345'
  title        TEXT    -- full trial name
  phase        TEXT    -- 'PHASE1', 'PHASE2', 'PHASE3', 'PHASE4', or NULL
                          multiple phases stored as 'PHASE1, PHASE2'
  condition    TEXT    -- disease/condition, e.g. 'Diabetes Mellitus, Type 2'
  status       TEXT    -- 'RECRUITING', 'COMPLETED', 'ACTIVE_NOT_RECRUITING'
  sponsor      TEXT    -- funding organisation
  source_field TEXT    -- 'eligibility_criteria', 'brief_summary',
                          'detailed_description'
  created_at   TIMESTAMPTZ

IMPORTANT QUERY RULES:
- condition, phase, and sponsor use partial matching: condition ILIKE '%Diabetes%'
- status is an exact value:  status = 'RECRUITING'
- Always use COUNT(DISTINCT nct_id) — never COUNT(*) — for trial counts
- Return at most 20 rows; add LIMIT 20 to every query that returns rows
- SELECT only — never UPDATE, DELETE, DROP, INSERT, or ALTER
"""

_SQL_SYSTEM_PROMPT = f"""You are a PostgreSQL expert. Write a single valid SQL SELECT query
to answer the user's question using the schema below. Return ONLY the SQL — no
explanation, no markdown fences, no semicolon at the end.

{_SQL_SCHEMA}"""

# Only allow SELECT — block any destructive keywords.
# Named _dangerous (lowercase) so tests can import and inspect it directly.
_dangerous = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

MAX_ROWS = 20


def _generate_sql(question: str) -> str:
    """Call GPT-4o to translate a natural-language question into SQL."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model       = "gpt-4o",
        messages    = [
            {"role": "system", "content": _SQL_SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ],
        temperature = 0,
        max_tokens  = 256,
    )
    return response.choices[0].message.content.strip()


def _run_sql(sql: str) -> list[dict]:
    """Execute a SQL query and return rows as a list of dicts."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            columns = [desc[0] for desc in cur.description]
            rows    = cur.fetchmany(MAX_ROWS)
            return [dict(zip(columns, row)) for row in rows]


def _format_rows(rows: list[dict]) -> str:
    """Turn a list of row dicts into a readable plain-text table."""
    if not rows:
        return "The query returned no results."

    # Use the keys of the first row as column headers
    headers = list(rows[0].keys())
    lines   = [" | ".join(headers)]
    lines.append("-" * len(lines[0]))
    for row in rows:
        lines.append(" | ".join(str(row.get(h, "")) for h in headers))

    suffix = f"\n({len(rows)} row{'s' if len(rows) != 1 else ''} returned)"
    return "\n".join(lines) + suffix


@tool
def sql_lookup(question: str) -> str:
    """
    Answer a structured question about the trial database by generating and
    running a SQL query against PostgreSQL.

    Use this for counts, aggregations, or filtered lists, for example:
      - "How many Phase 3 diabetes trials are currently recruiting?"
      - "Which sponsors have funded the most cardiac arrest trials?"
      - "List all completed hypertension trials."

    Do NOT use for questions about trial text or eligibility content —
    use rag_search for those.
    """
    log.info("sql_lookup_called", question=question[:80])

    try:
        sql = _generate_sql(question)
        log.info("sql_generated", sql=sql)

        if _dangerous.search(sql):
            log.warning("sql_blocked", sql=sql)
            return "SQL generation produced an unsafe query. Only SELECT is allowed."

        rows = _run_sql(sql)
        result = _format_rows(rows)
        log.info("sql_lookup_complete", n_rows=len(rows))
        return f"SQL: {sql}\n\n{result}"

    except Exception as exc:
        log.error("sql_lookup_failed", error=str(exc))
        return f"SQL lookup failed: {exc}"


# ---------------------------------------------------------------------------
# Exported list — imported by agent/graph.py
# ---------------------------------------------------------------------------

TOOLS = [rag_search, web_search, sql_lookup]
