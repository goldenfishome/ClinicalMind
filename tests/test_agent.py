"""Tests for agent/prompts.py and agent/rag.py."""

from unittest.mock import MagicMock, patch

import pytest

from agent.prompts import build_rag_context, RAG_SYSTEM_PROMPT, RAG_CONTEXT_TEMPLATE
from agent.rag     import ask, _extract_nct_ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_chunk(nct_id: str = "NCT00000001", score: float = 0.85) -> dict:
    return {
        "nct_id":       nct_id,
        "title":        "Test Diabetes Trial",
        "phase":        "PHASE3",
        "condition":    "Diabetes",
        "status":       "COMPLETED",
        "sponsor":      "NIH",
        "source_field": "eligibility_criteria",
        "section":      "Inclusion Criteria",
        "chunk_text":   "Patients must be 18 or older and diagnosed with Type 2 Diabetes.",
        "score":        score,
    }


def _mock_llm_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    return response


# ---------------------------------------------------------------------------
# agent/prompts.py
# ---------------------------------------------------------------------------

def test_build_rag_context_includes_nct_id():
    chunks = [_fake_chunk("NCT12345678")]
    context = build_rag_context(chunks)
    assert "NCT12345678" in context


def test_build_rag_context_numbers_excerpts():
    chunks = [_fake_chunk("NCT00000001"), _fake_chunk("NCT00000002")]
    context = build_rag_context(chunks)
    assert "[1]" in context
    assert "[2]" in context


def test_build_rag_context_includes_chunk_text():
    chunk = _fake_chunk()
    context = build_rag_context([chunk])
    assert chunk["chunk_text"] in context


def test_build_rag_context_includes_score():
    chunk = _fake_chunk(score=0.92)
    context = build_rag_context([chunk])
    assert "0.920" in context


def test_rag_system_prompt_contains_citation_rule():
    assert "NCT" in RAG_SYSTEM_PROMPT


def test_rag_system_prompt_contains_only_from_context_rule():
    assert "ONLY" in RAG_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# _extract_nct_ids
# ---------------------------------------------------------------------------

def test_extract_nct_ids_finds_ids():
    text = "The trial NCT04512345 showed improvement. See also NCT99887766."
    assert _extract_nct_ids(text) == ["NCT04512345", "NCT99887766"]


def test_extract_nct_ids_deduplicates():
    text = "NCT04512345 and NCT04512345 again."
    assert _extract_nct_ids(text) == ["NCT04512345"]


def test_extract_nct_ids_preserves_order():
    text = "NCT00000002 was first, then NCT00000001."
    assert _extract_nct_ids(text) == ["NCT00000002", "NCT00000001"]


def test_extract_nct_ids_returns_empty_for_no_ids():
    assert _extract_nct_ids("No trial IDs mentioned here.") == []


# ---------------------------------------------------------------------------
# ask()
# ---------------------------------------------------------------------------

def test_ask_returns_expected_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    answer_text = "Patients must be 18+ (NCT04512345)."

    with patch("agent.rag.search", return_value=[_fake_chunk()]):
        with patch("agent.rag.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = (
                _mock_llm_response(answer_text)
            )
            result = ask("What are the inclusion criteria?")

    assert "answer"      in result
    assert "sources"     in result
    assert "chunks_used" in result
    assert "error"       in result


def test_ask_extracts_sources_from_answer(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    answer_text = "Exclusion criteria include pregnancy (NCT04512345) and renal failure (NCT99887766)."

    with patch("agent.rag.search", return_value=[_fake_chunk()]):
        with patch("agent.rag.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = (
                _mock_llm_response(answer_text)
            )
            result = ask("What are exclusion criteria?")

    assert "NCT04512345" in result["sources"]
    assert "NCT99887766" in result["sources"]


def test_ask_returns_safe_message_when_no_chunks(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    with patch("agent.rag.search", return_value=[]):
        result = ask("Something obscure with no matches")

    assert result["error"] is None
    assert result["sources"] == []
    assert "could not find" in result["answer"].lower()


def test_ask_returns_safe_message_on_llm_error(monkeypatch):
    from openai import APIError

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    with patch("agent.rag.search", return_value=[_fake_chunk()]):
        with patch("agent.rag.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = (
                APIError("server error", request=MagicMock(), body={})
            )
            result = ask("What are the criteria?")

    assert result["error"] is not None
    assert "error" in result["answer"].lower()
    assert result["sources"] == []


def test_ask_passes_filters_to_search(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    answer_text = "Answer with no NCT IDs."

    with patch("agent.rag.search", return_value=[_fake_chunk()]) as mock_search:
        with patch("agent.rag.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = (
                _mock_llm_response(answer_text)
            )
            ask("query", condition="Diabetes", phase="PHASE3", status="RECRUITING")

    mock_search.assert_called_once_with(
        "query",
        top_k=5,
        condition="Diabetes",
        phase="PHASE3",
        status="RECRUITING",
    )


def test_ask_raises_on_empty_question(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with pytest.raises(ValueError, match="empty"):
        ask("   ")


def test_ask_uses_temperature_zero(monkeypatch):
    """Grounded answers require temperature=0 — verify it is always set."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")  # bypass wrap_openai

    with patch("agent.rag.search", return_value=[_fake_chunk()]):
        with patch("agent.rag.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            mock_client.chat.completions.create.return_value = (
                _mock_llm_response("Answer (NCT04512345).")
            )
            ask("What are the criteria?")

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["temperature"] == 0
