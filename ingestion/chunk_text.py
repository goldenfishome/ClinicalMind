"""
Section-based text chunker for parsed clinical trial records.

Splits the three free-text fields of a trial into focused chunks.
Each chunk keeps the full trial metadata so the retrieval layer can
filter and cite sources without joining back to another table.

Chunking strategy per field:
  - brief_summary        : single chunk (usually <300 tokens); split on
                           paragraphs only if it exceeds MAX_TOKENS.
  - eligibility_criteria : split on recognised section headers
                           (Inclusion / Exclusion / Key Criteria, etc.).
                           Each header + its bullets becomes one chunk.
  - detailed_description : split on paragraph boundaries; merge small
                           paragraphs until the chunk reaches MIN_TOKENS,
                           then flush.

Token counts use tiktoken (cl100k_base) — the same tokeniser as
text-embedding-3-small, so targets are accurate.

Usage:
    python3 -m ingestion.chunk_text
"""

import re
from typing import Generator

import tiktoken
import structlog

from agent.logging_config import setup_logging

setup_logging()
log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MIN_TOKENS = 50    # discard chunks shorter than this
MAX_TOKENS = 800   # flush a chunk when it reaches this size

EMBEDDING_ENCODING = "cl100k_base"

# Regex that matches common eligibility section headers.
# Examples:  "Inclusion Criteria:", "Key Exclusion Criteria:", "Other Criteria:"
_SECTION_HEADER_RE = re.compile(
    r"(?m)^[ \t]*"                          # optional leading whitespace
    r"((?:key\s+)?(?:inclusion|exclusion|eligibility|other)\s+criteria\s*:)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_enc = tiktoken.get_encoding(EMBEDDING_ENCODING)


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


# ---------------------------------------------------------------------------
# Low-level splitters
# ---------------------------------------------------------------------------

def _split_by_section_headers(text: str) -> list[tuple[str, str]]:
    """
    Split text on eligibility section headers.
    Returns [(section_label, section_text), ...].
    If no headers are found, returns [("", full_text)].
    """
    parts = _SECTION_HEADER_RE.split(text)
    # split() with a capturing group returns: [before, h1, body1, h2, body2, ...]
    if len(parts) <= 1:
        return [("", text.strip())]

    result: list[tuple[str, str]] = []
    # parts[0] is text before the first header (often empty or a preamble)
    preamble = parts[0].strip()
    if preamble:
        result.append(("", preamble))

    # Walk header + body pairs
    it = iter(parts[1:])
    for header in it:
        body = next(it, "").strip()
        label = header.strip().rstrip(":").strip()
        if body:
            result.append((label, f"{header.strip()}\n{body}"))

    return result or [("", text.strip())]


def _split_oversized_paragraph(text: str, max_tokens: int) -> list[str]:
    """
    Last-resort splitter for a single paragraph that has no blank-line breaks
    and exceeds max_tokens. Splits on sentence boundaries ('. ', '? ', '! ').
    """
    sentences = re.split(r"(?<=[.?!])\s+", text)
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for sentence in sentences:
        s_tokens = _count_tokens(sentence)
        if buffer_tokens + s_tokens > max_tokens and buffer:
            chunks.append(" ".join(buffer))
            buffer, buffer_tokens = [], 0
        buffer.append(sentence)
        buffer_tokens += s_tokens

    if buffer:
        chunks.append(" ".join(buffer))

    return [c for c in chunks if c.strip()]


def _split_by_paragraphs(text: str, min_tokens: int, max_tokens: int) -> list[str]:
    """
    Split on blank lines; merge short paragraphs to approach min_tokens;
    flush when the accumulated text reaches max_tokens.
    Paragraphs that individually exceed max_tokens are split at sentence
    boundaries as a fallback.
    """
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # Expand any oversized paragraphs before the merge pass
    paragraphs: list[str] = []
    for para in raw_paragraphs:
        if _count_tokens(para) > max_tokens:
            paragraphs.extend(_split_oversized_paragraph(para, max_tokens))
        else:
            paragraphs.append(para)

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for para in paragraphs:
        para_tokens = _count_tokens(para)

        if buffer_tokens + para_tokens > max_tokens and buffer:
            chunks.append("\n\n".join(buffer))
            buffer, buffer_tokens = [], 0

        buffer.append(para)
        buffer_tokens += para_tokens

        if buffer_tokens >= min_tokens:
            chunks.append("\n\n".join(buffer))
            buffer, buffer_tokens = [], 0

    if buffer:
        chunks.append("\n\n".join(buffer))

    return [c for c in chunks if _count_tokens(c) >= MIN_TOKENS]


# ---------------------------------------------------------------------------
# Per-field chunkers
# ---------------------------------------------------------------------------

def _chunks_for_summary(trial: dict) -> Generator[dict, None, None]:
    text = (trial.get("brief_summary") or "").strip()
    if not text:
        return

    if _count_tokens(text) <= MAX_TOKENS:
        yield _make_chunk(trial, "brief_summary", "", text)
    else:
        for i, part in enumerate(_split_by_paragraphs(text, MIN_TOKENS, MAX_TOKENS)):
            yield _make_chunk(trial, "brief_summary", f"Part {i + 1}", part)


def _chunks_for_eligibility(trial: dict) -> Generator[dict, None, None]:
    text = (trial.get("eligibility_criteria") or "").strip()
    if not text:
        return

    for section_label, section_text in _split_by_section_headers(text):
        if _count_tokens(section_text) < MIN_TOKENS:
            continue
        if _count_tokens(section_text) <= MAX_TOKENS:
            yield _make_chunk(trial, "eligibility_criteria", section_label, section_text)
        else:
            # Section is too long — fall back to paragraph splitting
            for i, part in enumerate(_split_by_paragraphs(section_text, MIN_TOKENS, MAX_TOKENS)):
                label = f"{section_label} (Part {i + 1})" if section_label else f"Part {i + 1}"
                yield _make_chunk(trial, "eligibility_criteria", label, part)


def _chunks_for_description(trial: dict) -> Generator[dict, None, None]:
    text = (trial.get("detailed_description") or "").strip()
    if not text:
        return

    for i, part in enumerate(_split_by_paragraphs(text, MIN_TOKENS, MAX_TOKENS)):
        label = f"Part {i + 1}" if i > 0 else ""
        yield _make_chunk(trial, "detailed_description", label, part)


# ---------------------------------------------------------------------------
# Chunk assembly
# ---------------------------------------------------------------------------

def _make_chunk(trial: dict, source_field: str, section: str, chunk_text: str) -> dict:
    return {
        "nct_id":       trial["nct_id"],
        "title":        trial.get("title", ""),
        "phase":        trial.get("phase"),
        "condition":    trial.get("condition"),
        "status":       trial.get("status"),
        "sponsor":      trial.get("sponsor"),
        "source_field": source_field,
        "section":      section,
        "chunk_text":   chunk_text,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_trial(trial: dict) -> list[dict]:
    """
    Chunk all three free-text fields of one parsed trial record.
    Returns a list of chunk dicts ready for embedding.
    """
    chunks = []
    chunks.extend(_chunks_for_summary(trial))
    chunks.extend(_chunks_for_eligibility(trial))
    chunks.extend(_chunks_for_description(trial))
    return chunks


def chunk_all(trials: list[dict]) -> list[dict]:
    """Chunk every trial in a list. Returns flat list of all chunks."""
    all_chunks: list[dict] = []
    for trial in trials:
        trial_chunks = chunk_trial(trial)
        all_chunks.extend(trial_chunks)
        log.debug("chunked_trial", nct_id=trial["nct_id"], chunks=len(trial_chunks))

    log.info("chunk_complete", trials=len(trials), total_chunks=len(all_chunks))
    return all_chunks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from ingestion.parse_trials import parse_all

    trials = parse_all()
    if not trials:
        print("No parsed trials found. Run ingestion.fetch_trials first.")
    else:
        chunks = chunk_all(trials)
        print(f"\nTotal chunks: {len(chunks)} from {len(trials)} trials")
        print(f"Avg chunks/trial: {len(chunks) / len(trials):.1f}")

        sample = chunks[0]
        print(f"\nSample chunk ({sample['nct_id']} — {sample['source_field']}):")
        for k, v in sample.items():
            if k == "chunk_text":
                print(f"  {'chunk_text':<15} {v[:200].replace(chr(10), ' ')}...")
            else:
                print(f"  {k:<15} {v}")
