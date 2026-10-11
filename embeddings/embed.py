"""
Generate embeddings for chunk dicts using OpenAI text-embedding-3-small.

Chunks are sent in batches to stay within API limits. Each chunk dict is
returned with an "embedding" key added (list of 1536 floats).

Handles:
  - Batching (BATCH_SIZE chunks per API call)
  - Rate-limit and transient errors (exponential backoff, max 3 retries)
  - Graceful logging — a failed batch is logged and skipped, not a crash

Usage:
    python3 -m embeddings.embed          # reads from parse+chunk pipeline
"""

import os
import time

import structlog
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError

from agent.logging_config import setup_logging

load_dotenv()
setup_logging()
log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS  = 1536
BATCH_SIZE      = 200   # texts per API call (OpenAI max is 2048)
MAX_RETRIES     = 3

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]] | None:
    """
    Embed one batch of texts with exponential backoff on rate limits.
    Returns list of embedding vectors, or None if all retries fail.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
            )
            return [item.embedding for item in response.data]

        except RateLimitError:
            wait = 2 ** attempt
            log.warning("rate_limit", attempt=attempt, wait_seconds=wait)
            time.sleep(wait)

        except APIError as exc:
            wait = 2 ** attempt
            log.warning("api_error", attempt=attempt, wait_seconds=wait, error=str(exc))
            time.sleep(wait)

    log.error("embed_batch_failed", batch_size=len(texts))
    return None


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add an "embedding" field to each chunk dict.

    Chunks that fail embedding (after retries) are dropped and logged.
    Returns only the chunks that were successfully embedded.
    """
    if not chunks:
        return []

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set in the environment.")

    client = OpenAI(api_key=api_key)
    embedded: list[dict] = []
    total   = len(chunks)

    for batch_start in range(0, total, BATCH_SIZE):
        batch        = chunks[batch_start : batch_start + BATCH_SIZE]
        texts        = [c["chunk_text"] for c in batch]
        batch_num    = batch_start // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        log.info(
            "embedding_batch",
            batch=f"{batch_num}/{total_batches}",
            chunks=len(batch),
        )

        vectors = _embed_batch(client, texts)
        if vectors is None:
            log.error("skipping_batch", batch_start=batch_start, size=len(batch))
            continue

        for chunk, vector in zip(batch, vectors):
            embedded.append({**chunk, "embedding": vector})

    dropped = total - len(embedded)
    if dropped:
        log.warning("embedding_dropped_chunks", dropped=dropped)

    log.info("embedding_complete", total=total, embedded=len(embedded))
    return embedded


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from pathlib import Path
    from ingestion.parse_trials import parse_all
    from ingestion.chunk_text   import chunk_all

    CACHE_PATH = Path("data/processed/embedded_chunks.json")

    trials = parse_all()
    if not trials:
        print("No parsed trials found. Run ingestion.fetch_trials first.")
    else:
        chunks   = chunk_all(trials)
        embedded = embed_chunks(chunks)
        # Save to disk so store.py doesn't need to re-embed
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(embedded))
        print(f"\nEmbedded {len(embedded)} chunks")
        print(f"Saved to {CACHE_PATH}")
        if embedded:
            sample = embedded[0]
            print(f"Sample: {sample['nct_id']} — {sample['source_field']}")
            print(f"  Embedding dims : {len(sample['embedding'])}")
            print(f"  First 5 values : {sample['embedding'][:5]}")
