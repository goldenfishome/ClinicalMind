"""
Insert embedded chunks into the PostgreSQL pgvector table.

Idempotency: uses ON CONFLICT (nct_id, source_field, section) DO UPDATE
so re-running the pipeline refreshes embeddings without creating duplicates.

Usage:
    python3 -m embeddings.store
"""

import structlog
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

from agent.logging_config import setup_logging
from database.connection import get_conn

setup_logging()
log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BATCH_SIZE = 500  # rows per INSERT statement

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

_UPSERT_SQL = """
INSERT INTO trial_chunks
    (nct_id, title, phase, condition, status, sponsor,
     source_field, section, chunk_text, embedding)
VALUES %s
ON CONFLICT (nct_id, source_field, section)
DO UPDATE SET
    title       = EXCLUDED.title,
    phase       = EXCLUDED.phase,
    condition   = EXCLUDED.condition,
    status      = EXCLUDED.status,
    sponsor     = EXCLUDED.sponsor,
    chunk_text  = EXCLUDED.chunk_text,
    embedding   = EXCLUDED.embedding,
    created_at  = NOW();
"""


def _truncate(value: str | None, max_len: int = 500) -> str | None:
    """Truncate long text fields that would overflow a btree index page."""
    if value and len(value) > max_len:
        return value[:max_len]
    return value


def _to_row(chunk: dict) -> tuple:
    """Convert a chunk dict to a tuple matching the INSERT column order."""
    return (
        chunk["nct_id"],
        _truncate(chunk.get("title", ""),    max_len=500),
        _truncate(chunk.get("phase"),        max_len=100),
        _truncate(chunk.get("condition"),    max_len=500),
        _truncate(chunk.get("status"),       max_len=100),
        _truncate(chunk.get("sponsor"),      max_len=500),
        chunk["source_field"],
        chunk.get("section") or "",   # NULL → empty string for unique constraint
        chunk["chunk_text"],
        chunk["embedding"],
    )


def store_chunks(chunks: list[dict]) -> int:
    """
    Upsert a list of embedded chunk dicts into trial_chunks.
    Returns the number of rows processed.
    """
    if not chunks:
        log.warning("store_chunks_empty")
        return 0

    # Validate all chunks have embeddings before touching the DB
    missing = [c["nct_id"] for c in chunks if "embedding" not in c]
    if missing:
        raise ValueError(
            f"{len(missing)} chunk(s) are missing embeddings "
            f"(first: {missing[0]}). Run embed_chunks() first."
        )

    # Deduplicate by unique constraint key (nct_id, source_field, section).
    # Raw data can contain duplicate section headers (e.g. "Inclusion Criteria"
    # appearing twice in one trial). PostgreSQL's ON CONFLICT DO UPDATE cannot
    # update the same row twice in a single INSERT — so we resolve duplicates
    # here before any batch reaches the database. Last occurrence wins.
    seen: dict[tuple, dict] = {}
    for chunk in chunks:
        key = (chunk["nct_id"], chunk["source_field"], chunk.get("section") or "")
        seen[key] = chunk
    original_count = len(chunks)
    chunks = list(seen.values())
    dupes = original_count - len(chunks)
    if dupes:
        log.warning("deduped_chunks", duplicates_removed=dupes)

    total_stored = 0

    with get_conn() as conn:
        register_vector(conn)  # teach psycopg2 how to serialise vector(1536)
        with conn.cursor() as cur:
            for batch_start in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[batch_start : batch_start + BATCH_SIZE]
                rows  = [_to_row(c) for c in batch]

                execute_values(cur, _UPSERT_SQL, rows)
                total_stored += len(batch)

                log.info(
                    "stored_batch",
                    rows=len(batch),
                    progress=f"{total_stored}/{len(chunks)}",
                )

    log.info("store_complete", total=total_stored)
    return total_stored


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from pathlib import Path

    CACHE_PATH = Path("data/processed/embedded_chunks.json")

    if not CACHE_PATH.exists():
        print(f"Embedded chunks cache not found at {CACHE_PATH}")
        print("Run  python3 -m embeddings.embed  first.")
    else:
        print(f"Loading embedded chunks from {CACHE_PATH} ...")
        embedded = json.loads(CACHE_PATH.read_text())
        print(f"Loaded {len(embedded):,} chunks — storing into PostgreSQL ...")
        stored = store_chunks(embedded)
        print(f"\nStored {stored:,} chunks into trial_chunks table.")
