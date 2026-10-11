"""Tests for retrieval/search.py."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from retrieval.search import search, _embed_query, DEFAULT_TOP_K


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_VECTOR = [0.01] * 1536

FAKE_ROW = (
    "NCT00000001",       # nct_id
    "Diabetes Trial",    # title
    "PHASE3",            # phase
    "Diabetes",          # condition
    "COMPLETED",         # status
    "NIH",               # sponsor
    "eligibility_criteria",  # source_field
    "Inclusion Criteria",    # section
    "Patients must be 18 or older and diagnosed with Type 2 Diabetes.",  # chunk_text
    0.87,                # score
)

FAKE_COLUMNS = [
    "nct_id", "title", "phase", "condition", "status",
    "sponsor", "source_field", "section", "chunk_text", "score",
]


def _mock_cursor(rows: list[tuple], columns: list[str]) -> MagicMock:
    cur = MagicMock()
    cur.fetchall.return_value = rows
    cur.description = [MagicMock(name="col") for col in columns]
    for mock_col, name in zip(cur.description, columns):
        mock_col.__getitem__ = lambda self, i, n=name: n if i == 0 else None
    # Make cur.description[i][0] return the column name
    cur.description = [(col,) for col in columns]
    return cur


def _mock_conn_ctx(cursor: MagicMock) -> MagicMock:
    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    @contextmanager
    def _ctx():
        yield conn

    return _ctx


# ---------------------------------------------------------------------------
# _embed_query
# ---------------------------------------------------------------------------

def test_embed_query_returns_vector(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch("retrieval.search.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=FAKE_VECTOR)]
        )
        result = _embed_query("What are exclusion criteria?")

    assert result == FAKE_VECTOR
    mock_client.embeddings.create.assert_called_once()


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def test_search_returns_list_of_dicts(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cursor = _mock_cursor([FAKE_ROW], FAKE_COLUMNS)

    with patch("retrieval.search._embed_query", return_value=FAKE_VECTOR):
        with patch("retrieval.search.get_conn", _mock_conn_ctx(cursor)):
            with patch("retrieval.search.register_vector"):
                results = search("What are the inclusion criteria?")

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["nct_id"] == "NCT00000001"
    assert results[0]["score"] == 0.87


def test_search_result_has_all_fields(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cursor = _mock_cursor([FAKE_ROW], FAKE_COLUMNS)

    with patch("retrieval.search._embed_query", return_value=FAKE_VECTOR):
        with patch("retrieval.search.get_conn", _mock_conn_ctx(cursor)):
            with patch("retrieval.search.register_vector"):
                results = search("diabetes exclusion criteria")

    result = results[0]
    for field in ["nct_id", "title", "phase", "condition", "status",
                  "source_field", "section", "chunk_text", "score"]:
        assert field in result, f"Missing field: {field}"


def test_search_passes_top_k_to_query(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cursor = _mock_cursor([], FAKE_COLUMNS)

    with patch("retrieval.search._embed_query", return_value=FAKE_VECTOR):
        with patch("retrieval.search.get_conn", _mock_conn_ctx(cursor)):
            with patch("retrieval.search.register_vector"):
                search("query", top_k=10)

    # top_k=10 should be the last param passed to cursor.execute
    call_args = cursor.execute.call_args
    params = call_args[0][1]   # second positional arg to execute() is the params list
    assert params[-1] == 10


def test_search_includes_condition_filter(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cursor = _mock_cursor([], FAKE_COLUMNS)

    with patch("retrieval.search._embed_query", return_value=FAKE_VECTOR):
        with patch("retrieval.search.get_conn", _mock_conn_ctx(cursor)):
            with patch("retrieval.search.register_vector"):
                search("query", condition="Diabetes")

    sql = cursor.execute.call_args[0][0]
    assert "condition ILIKE" in sql


def test_search_includes_phase_filter(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cursor = _mock_cursor([], FAKE_COLUMNS)

    with patch("retrieval.search._embed_query", return_value=FAKE_VECTOR):
        with patch("retrieval.search.get_conn", _mock_conn_ctx(cursor)):
            with patch("retrieval.search.register_vector"):
                search("query", phase="PHASE3")

    sql = cursor.execute.call_args[0][0]
    assert "phase ILIKE" in sql


def test_search_includes_status_filter(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cursor = _mock_cursor([], FAKE_COLUMNS)

    with patch("retrieval.search._embed_query", return_value=FAKE_VECTOR):
        with patch("retrieval.search.get_conn", _mock_conn_ctx(cursor)):
            with patch("retrieval.search.register_vector"):
                search("query", status="RECRUITING")

    sql = cursor.execute.call_args[0][0]
    assert "status ILIKE" in sql


def test_search_no_filters_has_no_where_clause(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cursor = _mock_cursor([], FAKE_COLUMNS)

    with patch("retrieval.search._embed_query", return_value=FAKE_VECTOR):
        with patch("retrieval.search.get_conn", _mock_conn_ctx(cursor)):
            with patch("retrieval.search.register_vector"):
                search("query")

    sql = cursor.execute.call_args[0][0]
    assert "WHERE" not in sql


def test_search_raises_on_empty_query(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with pytest.raises(ValueError, match="empty"):
        search("   ")


def test_search_returns_empty_list_when_no_results(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cursor = _mock_cursor([], FAKE_COLUMNS)

    with patch("retrieval.search._embed_query", return_value=FAKE_VECTOR):
        with patch("retrieval.search.get_conn", _mock_conn_ctx(cursor)):
            with patch("retrieval.search.register_vector"):
                results = search("obscure query with no matches")

    assert results == []
