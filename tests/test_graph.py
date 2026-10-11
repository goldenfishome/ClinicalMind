"""Tests for agent/graph.py — the LangGraph ReAct agent skeleton."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import (
    AgentState,
    _should_continue,
    agent_node,
    run_agent,
    tools_node,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(messages, sources=None) -> AgentState:
    return AgentState(
        messages=messages,
        question=messages[0].content if messages else "",
        sources=sources or [],
    )


def _ai_no_tools(content: str = "Final answer.") -> AIMessage:
    """AIMessage with no tool calls — signals end of loop."""
    return AIMessage(content=content)


def _ai_with_tool(tool_name: str, args: dict, tc_id: str = "call_001") -> AIMessage:
    """AIMessage that requests one tool call."""
    msg = AIMessage(content="")
    msg.tool_calls = [{"name": tool_name, "args": args, "id": tc_id}]
    return msg


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------

def test_agent_state_has_required_keys():
    state = _state([HumanMessage(content="Hello")])
    assert "messages"  in state
    assert "question"  in state
    assert "sources"   in state


def test_agent_state_sources_defaults_to_empty():
    state = AgentState(
        messages=[HumanMessage(content="test")],
        question="test",
        sources=[],
    )
    assert state["sources"] == []


# ---------------------------------------------------------------------------
# _should_continue routing
# ---------------------------------------------------------------------------

def test_should_continue_returns_tools_when_tool_calls_present():
    ai_msg = _ai_with_tool("rag_search", {"query": "diabetes"})
    state  = _state([HumanMessage(content="q"), ai_msg])
    assert _should_continue(state) == "tools"


def test_should_continue_returns_end_when_no_tool_calls():
    state = _state([HumanMessage(content="q"), _ai_no_tools("Done.")])
    assert _should_continue(state) == "end"


def test_should_continue_returns_end_for_empty_tool_calls():
    ai_msg = AIMessage(content="Final answer.")
    ai_msg.tool_calls = []
    state = _state([HumanMessage(content="q"), ai_msg])
    assert _should_continue(state) == "end"


# ---------------------------------------------------------------------------
# tools_node
# ---------------------------------------------------------------------------

def test_tools_node_returns_tool_message():
    ai_msg = _ai_with_tool("rag_search", {"query": "diabetes"}, tc_id="c1")
    state  = _state([HumanMessage(content="q"), ai_msg])
    result = tools_node(state)
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], ToolMessage)
    assert result["messages"][0].tool_call_id == "c1"


def test_tools_node_handles_unknown_tool_gracefully():
    ai_msg = _ai_with_tool("nonexistent_tool", {}, tc_id="c2")
    state  = _state([HumanMessage(content="q"), ai_msg])
    result = tools_node(state)
    assert "unknown tool" in result["messages"][0].content


def test_tools_node_extracts_nct_ids_from_output():
    """NCT IDs in tool output should be added to sources."""
    from agent import graph as g
    # StructuredTool is a frozen Pydantic model — patch the _TOOL_MAP dict entry
    # rather than trying to set an attribute on the tool object directly.
    mock_tool = MagicMock()
    mock_tool.invoke.return_value = "Trial NCT04512345 showed improvement."
    with patch.dict(g._TOOL_MAP, {"rag_search": mock_tool}):
        ai_msg = _ai_with_tool("rag_search", {"query": "test"}, tc_id="c3")
        state  = _state([HumanMessage(content="q"), ai_msg])
        result = tools_node(state)
    assert "NCT04512345" in result["sources"]


def test_tools_node_handles_multiple_tool_calls():
    ai_msg = AIMessage(content="")
    ai_msg.tool_calls = [
        {"name": "rag_search", "args": {"query": "diabetes"}, "id": "c4"},
        {"name": "sql_lookup", "args": {"question": "count"}, "id": "c5"},
    ]
    state  = _state([HumanMessage(content="q"), ai_msg])
    result = tools_node(state)
    assert len(result["messages"]) == 2


# ---------------------------------------------------------------------------
# agent_node
# ---------------------------------------------------------------------------

def _mock_llm_response(content: str, tool_calls=None) -> AIMessage:
    msg = AIMessage(content=content)
    msg.tool_calls = tool_calls or []
    return msg


def test_agent_node_returns_ai_message(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    with patch("agent.graph.ChatOpenAI") as mock_cls:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = _mock_llm_response("Here is the answer.")
        mock_cls.return_value = mock_llm

        state  = _state([HumanMessage(content="What are diabetes exclusion criteria?")])
        result = agent_node(state)

    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)


def test_agent_node_collects_nct_ids_from_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    with patch("agent.graph.ChatOpenAI") as mock_cls:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = _mock_llm_response(
            "See NCT04512345 and NCT99887766 for details."
        )
        mock_cls.return_value = mock_llm

        state  = _state([HumanMessage(content="test")])
        result = agent_node(state)

    assert "NCT04512345" in result["sources"]
    assert "NCT99887766" in result["sources"]


def test_agent_node_deduplicates_sources(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    with patch("agent.graph.ChatOpenAI") as mock_cls:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = _mock_llm_response(
            "NCT04512345 and NCT04512345 again."
        )
        mock_cls.return_value = mock_llm

        state  = _state([HumanMessage(content="test")], sources=["NCT04512345"])
        result = agent_node(state)

    assert result["sources"].count("NCT04512345") == 1


# ---------------------------------------------------------------------------
# run_agent — end-to-end with mocked LLM
# ---------------------------------------------------------------------------

def test_run_agent_returns_expected_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    with patch("agent.graph.ChatOpenAI") as mock_cls:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = _mock_llm_response("Final answer (NCT04512345).")
        mock_cls.return_value = mock_llm

        result = run_agent("What are the exclusion criteria?")

    assert "answer"   in result
    assert "sources"  in result
    assert "messages" in result
    assert "error"    in result


def test_run_agent_raises_on_empty_question(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with pytest.raises(ValueError, match="empty"):
        run_agent("   ")


def test_run_agent_returns_safe_message_on_exception(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    with patch("agent.graph.agent_graph") as mock_graph:
        mock_graph.invoke.side_effect = RuntimeError("DB connection lost")
        result = run_agent("What are the criteria?")

    assert result["error"] is not None
    assert "error" in result["answer"].lower()
    assert result["sources"] == []


def test_run_agent_single_turn_no_tools(monkeypatch):
    """When the LLM answers directly with no tool calls, the loop ends after one turn."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    with patch("agent.graph.ChatOpenAI") as mock_cls:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = _mock_llm_response("Direct answer, no tools needed.")
        mock_cls.return_value = mock_llm

        result = run_agent("Hello")

    assert result["answer"] == "Direct answer, no tools needed."
    assert result["error"] is None
