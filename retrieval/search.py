"""
RAG retrieval: embed a query and find the most similar chunks in pgvector.

Supports optional metadata pre-filtering by condition, phase, and status
before the cosine similarity search runs — this narrows the search space
and improves precision for targeted questions.

Usage:
    python3 -m retrieval.search
"""

import os

import structlog
from dotenv import load_dotenv
from langsmith import traceable
from openai import OpenAI
from pgvector.psycopg2 import register_vector

from agent.logging_config import setup_logging
from database.connection import get_conn

load_dotenv()
setup_logging()
log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TOP_K   = 5

# ---------------------------------------------------------------------------
# Query embedding
# ---------------------------------------------------------------------------

def _embed_query(query: str) -> list[float]:
    """Embed a single query string using the same model as the stored chunks."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@traceable(name="pgvector_search", run_type="retriever")
def search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    condition: str | None = None,
    phase: str | None = None,
    status: str | None = None,
) -> list[dict]:
    """
    Find the top-k most semantically similar chunks to the query.

    Optional filters narrow the search before similarity is computed:
      - condition : e.g. "Diabetes"  (case-insensitive substring match)
      - phase     : e.g. "PHASE3"    (case-insensitive substring match)
      - status    : e.g. "RECRUITING" (case-insensitive exact match)

    Returns a list of dicts, each containing chunk metadata and a
    similarity score (0–1, higher = more similar).
    """
    if not query.strip():
        raise ValueError("Query must not be empty.")

    query_vector = _embed_query(query)

    # Build WHERE clause dynamically from whichever filters were provided
    filters: list[str] = []
    params:  list      = []

    if condition:
        filters.append("condition ILIKE %s")
        params.append(f"%{condition}%")
    if phase:
        filters.append("phase ILIKE %s")
        params.append(f"%{phase}%")
    if status:
        filters.append("status ILIKE %s")
        params.append(f"%{status}%")

    where_clause = ("WHERE " + " AND ".join(filters)) if filters else ""

    sql = f"""
        SELECT
            nct_id,
            title,
            phase,
            condition,
            status,
            sponsor,
            source_field,
            section,
            chunk_text,
            1 - (embedding <=> %s::vector) AS score
        FROM trial_chunks
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """

    # query_vector appears twice: once for the score column, once for ORDER BY
    query_params = [query_vector] + params + [query_vector, top_k]

    with get_conn() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(sql, query_params)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]

    results = [dict(zip(columns, row)) for row in rows]

    log.info(
        "search_complete",
        query=query[:80],
        results=len(results),
        filters={"condition": condition, "phase": phase, "status": status},
    )

    return results


# ---------------------------------------------------------------------------
# Entry point — smoke test against a live database
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_queries = [
        ("What are exclusion criteria for diabetes trials?",   {"condition": "diabetes"}),
        ("How many patients were enrolled in Phase 3 trials?", {"phase": "PHASE3"}),
        ("What interventions are used for hypertension?",      {}),
    ]

    for query, filters in test_queries:
        print(f"\nQuery : {query}")
        print(f"Filter: {filters}")
        results = search(query, top_k=3, **filters)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] {r['nct_id']} | score={r['score']:.3f} | "
                  f"{r['source_field']} | {r['chunk_text'][:120].replace(chr(10),' ')}...")
