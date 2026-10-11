"""Tests for agent/tools.py — RAG Search, Web Search, SQL Lookup tools."""

from unittest.mock import MagicMock, patch

import pytest

from agent.tools import (
    TOOLS,
    _dangerous,
    _format_rows,
    _generate_sql,
    _is_placeholder_key,
    rag_search,
    sql_lookup,
    web_search,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ask_result(
    answer: str = "Patients must be 18+ (NCT04512345).",
    sources: list | None = None,
    chunks_used: list | None = None,
    error: str | None = None,
) -> dict:
    return {
        "answer":      answer,
        "sources":     sources if sources is not None else ["NCT04512345"],
        "chunks_used": chunks_used if chunks_used is not None else [{"chunk_text": "..."}],
        "error":       error,
    }


# ---------------------------------------------------------------------------
# TOOLS list
# ---------------------------------------------------------------------------

def test_tools_list_contains_three_tools():
    assert len(TOOLS) == 3


def test_tools_list_has_correct_names():
    names = {t.name for t in TOOLS}
    assert names == {"rag_search", "web_search", "sql_lookup"}


# ---------------------------------------------------------------------------
# rag_search
# ---------------------------------------------------------------------------

def test_rag_search_returns_answer_text():
    with patch("agent.tools.ask", return_value=_ask_result()):
        result = rag_search.invoke({"query": "What are diabetes exclusion criteria?"})
    assert "Patients must be 18+" in result


def test_rag_search_appends_sources_line():
    with patch("agent.tools.ask", return_value=_ask_result(sources=["NCT04512345"])):
        result = rag_search.invoke({"query": "test"})
    assert "Trials cited" in result
    assert "NCT04512345" in result


def test_rag_search_passes_filters_to_ask():
    with patch("agent.tools.ask", return_value=_ask_result()) as mock_ask:
        rag_search.invoke({
            "query": "criteria", "condition": "Diabetes",
            "phase": "PHASE3",   "status": "RECRUITING",
        })
    kw = mock_ask.call_args[1]
    assert kw["condition"] == "Diabetes"
    assert kw["phase"]     == "PHASE3"
    assert kw["status"]    == "RECRUITING"
    assert kw["top_k"]     == 5


def test_rag_search_filters_default_to_none():
    with patch("agent.tools.ask", return_value=_ask_result()) as mock_ask:
        rag_search.invoke({"query": "test"})
    kw = mock_ask.call_args[1]
    assert kw["condition"] is None
    assert kw["phase"]     is None
    assert kw["status"]    is None


def test_rag_search_handles_no_chunks():
    with patch("agent.tools.ask", return_value=_ask_result(chunks_used=[], sources=[])):
        result = rag_search.invoke({"query": "obscure"})
    assert "No relevant" in result


def test_rag_search_handles_llm_error():
    with patch("agent.tools.ask", return_value=_ask_result(error="timeout")):
        result = rag_search.invoke({"query": "test"})
    assert "error" in result.lower()


def test_rag_search_no_sources_cited_message():
    with patch("agent.tools.ask", return_value=_ask_result(sources=[])):
        result = rag_search.invoke({"query": "test"})
    assert "No NCT IDs" in result


# ---------------------------------------------------------------------------
# web_search — _is_placeholder_key helper
# ---------------------------------------------------------------------------

def test_placeholder_key_detected_for_none():
    assert _is_placeholder_key(None) is True


def test_placeholder_key_detected_for_empty():
    assert _is_placeholder_key("") is True


def test_placeholder_key_detected_for_example_value():
    assert _is_placeholder_key("tvly-...") is True


def test_real_key_not_flagged_as_placeholder():
    assert _is_placeholder_key("tvly-abc123realkey") is False


# ---------------------------------------------------------------------------
# web_search — missing / placeholder key
# ---------------------------------------------------------------------------

def test_web_search_returns_config_message_when_key_missing(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-...")
    result = web_search.invoke({"query": "latest FDA approvals"})
    assert "not configured" in result.lower() or "not available" in result.lower()


def test_web_search_does_not_raise_when_key_missing(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-...")
    result = web_search.invoke({"query": "anything"})
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# web_search — real key (mocked Tavily client)
# ---------------------------------------------------------------------------

def _tavily_response(answer="GPT-4o approved.", results=None):
    return {
        "answer":  answer,
        "results": results or [
            {"title": "FDA News", "url": "https://fda.gov", "content": "New approval…"},
        ],
    }


def test_web_search_returns_answer_when_key_set(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-realkey123")
    with patch("agent.tools.TavilyClient") as mock_cls:
        mock_cls.return_value.search.return_value = _tavily_response()
        result = web_search.invoke({"query": "latest FDA approvals"})
    assert "GPT-4o approved." in result


def test_web_search_includes_source_urls(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-realkey123")
    with patch("agent.tools.TavilyClient") as mock_cls:
        mock_cls.return_value.search.return_value = _tavily_response()
        result = web_search.invoke({"query": "test"})
    assert "https://fda.gov" in result


def test_web_search_handles_tavily_exception(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-realkey123")
    with patch("agent.tools.TavilyClient") as mock_cls:
        mock_cls.return_value.search.side_effect = Exception("network error")
        result = web_search.invoke({"query": "test"})
    assert "failed" in result.lower()


def test_web_search_handles_empty_results(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-realkey123")
    with patch("agent.tools.TavilyClient") as mock_cls:
        mock_cls.return_value.search.return_value = {"answer": "", "results": []}
        result = web_search.invoke({"query": "test"})
    assert "no results" in result.lower()


# ---------------------------------------------------------------------------
# sql_lookup — _is_dangerous guard
# ---------------------------------------------------------------------------

def test_dangerous_blocks_drop():
    assert _dangerous.search("DROP TABLE trial_chunks") is not None


def test_dangerous_blocks_delete():
    assert _dangerous.search("DELETE FROM trial_chunks") is not None


def test_dangerous_blocks_insert():
    assert _dangerous.search("INSERT INTO trial_chunks VALUES (1)") is not None


def test_dangerous_blocks_update():
    assert _dangerous.search("UPDATE trial_chunks SET title='x'") is not None


def test_dangerous_allows_select():
    assert _dangerous.search("SELECT COUNT(DISTINCT nct_id) FROM trial_chunks") is None


# ---------------------------------------------------------------------------
# sql_lookup — _format_rows helper
# ---------------------------------------------------------------------------

def test_format_rows_returns_no_results_for_empty():
    assert "no results" in _format_rows([]).lower()


def test_format_rows_includes_headers():
    rows = [{"nct_id": "NCT001", "count": 5}]
    out  = _format_rows(rows)
    assert "nct_id" in out
    assert "count"  in out


def test_format_rows_includes_values():
    rows = [{"nct_id": "NCT001", "count": 5}]
    out  = _format_rows(rows)
    assert "NCT001" in out
    assert "5"      in out


def test_format_rows_shows_row_count():
    rows = [{"nct_id": f"NCT{i:03d}"} for i in range(3)]
    out  = _format_rows(rows)
    assert "3 rows" in out


def test_format_rows_singular_for_one_row():
    rows = [{"nct_id": "NCT001"}]
    out  = _format_rows(rows)
    assert "1 row" in out


# ---------------------------------------------------------------------------
# sql_lookup — _generate_sql (mocked LLM)
# ---------------------------------------------------------------------------

def _mock_openai_response(sql: str):
    mock = MagicMock()
    mock.choices[0].message.content = sql
    return mock


def test_generate_sql_returns_string(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch("agent.tools.OpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create.return_value = (
            _mock_openai_response("SELECT COUNT(DISTINCT nct_id) FROM trial_chunks")
        )
        sql = _generate_sql("How many trials are there?")
    assert "SELECT" in sql.upper()


# ---------------------------------------------------------------------------
# sql_lookup — end-to-end (mocked LLM + DB)
# ---------------------------------------------------------------------------

def test_sql_lookup_returns_formatted_result(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    generated_sql = "SELECT COUNT(DISTINCT nct_id) AS total FROM trial_chunks"

    with patch("agent.tools._generate_sql", return_value=generated_sql):
        with patch("agent.tools._run_sql", return_value=[{"total": 42}]):
            result = sql_lookup.invoke({"question": "How many trials?"})

    assert "42"    in result
    assert "total" in result
    assert generated_sql in result   # SQL echoed at top


def test_sql_lookup_blocks_dangerous_sql(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch("agent.tools._generate_sql", return_value="DROP TABLE trial_chunks"):
        result = sql_lookup.invoke({"question": "delete everything"})
    assert "unsafe" in result.lower()


def test_sql_lookup_handles_db_exception(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch("agent.tools._generate_sql", return_value="SELECT 1 FROM trial_chunks"):
        with patch("agent.tools._run_sql", side_effect=Exception("connection refused")):
            result = sql_lookup.invoke({"question": "test"})
    assert "failed" in result.lower()


def test_sql_lookup_handles_zero_rows(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch("agent.tools._generate_sql", return_value="SELECT nct_id FROM trial_chunks WHERE nct_id='X'"):
        with patch("agent.tools._run_sql", return_value=[]):
            result = sql_lookup.invoke({"question": "find trial X"})
    assert "no results" in result.lower()
