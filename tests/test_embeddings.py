"""Tests for embeddings/embed.py."""

from unittest.mock import MagicMock, patch, call

import pytest
from openai import RateLimitError, APIError

from embeddings.embed import embed_chunks, _embed_batch, EMBEDDING_DIMS, BATCH_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_chunks(n: int) -> list[dict]:
    return [
        {
            "nct_id":       f"NCT{i:08d}",
            "title":        "Test Trial",
            "phase":        "PHASE3",
            "condition":    "Diabetes",
            "status":       "COMPLETED",
            "sponsor":      "NIH",
            "source_field": "brief_summary",
            "section":      "",
            "chunk_text":   f"This is chunk number {i} with some meaningful text.",
        }
        for i in range(n)
    ]


def _fake_embedding() -> list[float]:
    return [0.01] * EMBEDDING_DIMS


def _mock_openai_response(n: int) -> MagicMock:
    """Return a mock openai embeddings response for n texts."""
    response = MagicMock()
    response.data = [MagicMock(embedding=_fake_embedding()) for _ in range(n)]
    return response


# ---------------------------------------------------------------------------
# _embed_batch
# ---------------------------------------------------------------------------

def test_embed_batch_returns_vectors():
    client = MagicMock()
    client.embeddings.create.return_value = _mock_openai_response(3)

    vectors = _embed_batch(client, ["text1", "text2", "text3"])

    assert vectors is not None
    assert len(vectors) == 3
    assert len(vectors[0]) == EMBEDDING_DIMS


def test_embed_batch_retries_on_rate_limit():
    client = MagicMock()
    # First call raises RateLimitError, second succeeds
    client.embeddings.create.side_effect = [
        RateLimitError("rate limited", response=MagicMock(), body={}),
        _mock_openai_response(1),
    ]

    with patch("embeddings.embed.time.sleep"):
        vectors = _embed_batch(client, ["text"])

    assert vectors is not None
    assert client.embeddings.create.call_count == 2


def test_embed_batch_returns_none_after_max_retries():
    client = MagicMock()
    client.embeddings.create.side_effect = RateLimitError(
        "rate limited", response=MagicMock(), body={}
    )

    with patch("embeddings.embed.time.sleep"):
        result = _embed_batch(client, ["text"])

    assert result is None


def test_embed_batch_retries_on_api_error():
    client = MagicMock()
    client.embeddings.create.side_effect = [
        APIError("server error", request=MagicMock(), body={}),
        _mock_openai_response(1),
    ]

    with patch("embeddings.embed.time.sleep"):
        vectors = _embed_batch(client, ["text"])

    assert vectors is not None


# ---------------------------------------------------------------------------
# embed_chunks
# ---------------------------------------------------------------------------

def test_embed_chunks_adds_embedding_field(monkeypatch):
    chunks = _fake_chunks(3)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch("embeddings.embed.OpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        mock_client.embeddings.create.return_value = _mock_openai_response(3)

        result = embed_chunks(chunks)

    assert len(result) == 3
    for item in result:
        assert "embedding" in item
        assert len(item["embedding"]) == EMBEDDING_DIMS


def test_embed_chunks_preserves_metadata(monkeypatch):
    chunks = _fake_chunks(2)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch("embeddings.embed.OpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        mock_client.embeddings.create.return_value = _mock_openai_response(2)

        result = embed_chunks(chunks)

    for original, embedded in zip(chunks, result):
        assert embedded["nct_id"]       == original["nct_id"]
        assert embedded["source_field"] == original["source_field"]
        assert embedded["chunk_text"]   == original["chunk_text"]


def test_embed_chunks_batches_correctly(monkeypatch):
    """More chunks than BATCH_SIZE should trigger multiple API calls."""
    n = BATCH_SIZE + 5
    chunks = _fake_chunks(n)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch("embeddings.embed.OpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        mock_client.embeddings.create.side_effect = [
            _mock_openai_response(BATCH_SIZE),
            _mock_openai_response(5),
        ]

        result = embed_chunks(chunks)

    assert mock_client.embeddings.create.call_count == 2
    assert len(result) == n


def test_embed_chunks_skips_failed_batch(monkeypatch):
    """A batch that fails all retries should be dropped, not crash."""
    chunks = _fake_chunks(BATCH_SIZE + 5)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch("embeddings.embed.OpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        # First batch fails all retries, second batch succeeds
        mock_client.embeddings.create.side_effect = (
            [RateLimitError("rate limited", response=MagicMock(), body={})] * 3
            + [_mock_openai_response(5)]
        )

        with patch("embeddings.embed.time.sleep"):
            result = embed_chunks(chunks)

    # Only the second batch (5 chunks) should be in the result
    assert len(result) == 5


def test_embed_chunks_returns_empty_for_no_input(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert embed_chunks([]) == []


def test_embed_chunks_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
        embed_chunks(_fake_chunks(1))
