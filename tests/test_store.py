"""Tests for embeddings/store.py."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch, call

import pytest

from embeddings.store import store_chunks, _to_row, BATCH_SIZE
from embeddings.embed  import EMBEDDING_DIMS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_embedded_chunk(nct_id: str = "NCT00000001", section: str = "") -> dict:
    return {
        "nct_id":       nct_id,
        "title":        "Test Trial",
        "phase":        "PHASE3",
        "condition":    "Diabetes",
        "status":       "COMPLETED",
        "sponsor":      "NIH",
        "source_field": "brief_summary",
        "section":      section,
        "chunk_text":   "Some chunk text for testing purposes.",
        "embedding":    [0.01] * EMBEDDING_DIMS,
    }


def _mock_conn_ctx(mock_cursor: MagicMock):
    """Return a context manager that yields a mock psycopg2 connection."""
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    @contextmanager
    def _ctx():
        yield mock_conn

    return _ctx


# ---------------------------------------------------------------------------
# _to_row
# ---------------------------------------------------------------------------

def test_to_row_correct_order():
    chunk = _fake_embedded_chunk()
    row = _to_row(chunk)
    assert row[0] == "NCT00000001"   # nct_id
    assert row[6] == "brief_summary" # source_field
    assert row[8] == chunk["chunk_text"]
    assert row[9] == chunk["embedding"]


def test_to_row_null_section_becomes_empty_string():
    chunk = _fake_embedded_chunk()
    chunk["section"] = None
    row = _to_row(chunk)
    assert row[7] == ""


# ---------------------------------------------------------------------------
# store_chunks
# ---------------------------------------------------------------------------

def test_store_chunks_returns_count():
    chunks = [_fake_embedded_chunk(f"NCT{i:08d}") for i in range(3)]
    mock_cur = MagicMock()

    with patch("embeddings.store.get_conn", _mock_conn_ctx(mock_cur)):
        with patch("embeddings.store.register_vector"):
            with patch("embeddings.store.execute_values"):
                result = store_chunks(chunks)

    assert result == 3


def test_store_chunks_calls_execute_values():
    chunks = [_fake_embedded_chunk()]
    mock_cur = MagicMock()

    with patch("embeddings.store.get_conn", _mock_conn_ctx(mock_cur)):
        with patch("embeddings.store.register_vector"):
            with patch("embeddings.store.execute_values") as mock_ev:
                store_chunks(chunks)

    assert mock_ev.called


def test_store_chunks_batches_large_input():
    """Input larger than BATCH_SIZE should trigger multiple execute_values calls."""
    chunks = [_fake_embedded_chunk(f"NCT{i:08d}", section=f"part{i}") for i in range(BATCH_SIZE + 10)]
    mock_cur = MagicMock()

    with patch("embeddings.store.get_conn", _mock_conn_ctx(mock_cur)):
        with patch("embeddings.store.register_vector"):
            with patch("embeddings.store.execute_values") as mock_ev:
                store_chunks(chunks)

    assert mock_ev.call_count == 2


def test_store_chunks_raises_if_embedding_missing():
    chunk = _fake_embedded_chunk()
    del chunk["embedding"]

    with pytest.raises(ValueError, match="missing embeddings"):
        store_chunks([chunk])


def test_store_chunks_returns_zero_for_empty_input():
    with patch("embeddings.store.get_conn"):
        result = store_chunks([])
    assert result == 0


def test_store_chunks_registers_vector_type():
    """register_vector must be called so psycopg2 can serialise vector(1536)."""
    chunks = [_fake_embedded_chunk()]
    mock_cur = MagicMock()

    with patch("embeddings.store.get_conn", _mock_conn_ctx(mock_cur)):
        with patch("embeddings.store.register_vector") as mock_rv:
            with patch("embeddings.store.execute_values"):
                store_chunks(chunks)

    assert mock_rv.called
