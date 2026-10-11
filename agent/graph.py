"""
LangGraph ReAct agent for ClinicalMind.

Architecture
------------
The agent follows the standard ReAct (Reason + Act) loop:

    START
      │
      ▼
  ┌─────────┐     has tool_calls?     ┌───────────┐
  │  agent  │ ─────────── YES ──────▶ │   tools   │
  │  (LLM)  │                         │ (executor)│
  └─────────┘ ◀────────────────────── └───────────┘
      │
      │  no tool_calls → final answer
      ▼
     END

State
-----
AgentState is a TypedDict that accumulates across every node visit:
  - messages   : full conversation history (HumanMessage, AIMessage, ToolMessage)
  - question   : the original user question (never mutated)
  - sources    : NCT IDs cited anywhere in the conversation

Tools (agent/tools.py)
----------------------
  rag_search  : semantic search over pgvector chunks  ✅ Feature 13
  web_search  : Tavily web search                     🔧 Feature 14
  sql_lookup  : text-to-SQL over PostgreSQL           🔧 Feature 15

Usage:
    python3 -m agent.graph
"""

import os
import re
import time
from typing import Annotated

import structlog
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langsmith import traceable
from typing_extensions import TypedDict

from agent.logging_config import setup_logging
from agent.prompts import AGENT_SYSTEM_PROMPT
from agent.tools import TOOLS

load_dotenv()
setup_logging()
log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LLM_MODEL    = "gpt-4o"
_NCT_PATTERN = re.compile(r"NCT\d{8}")

# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    Shared state passed between every node in the graph.

    messages  : conversation history; the `add_messages` reducer appends new
                messages rather than overwriting — so each node only needs to
                return the *new* messages it produces.
    question  : the original user question, stored once at graph entry.
    sources   : deduplicated NCT IDs cited across all tool results and
                AI responses so far in this run.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    sources:  list[str]


# TOOLS imported from agent/tools.py (see that file for implementations).
# graph.py owns the loop; tools.py owns what each tool does.

# ---------------------------------------------------------------------------
# Node: agent (LLM with tools bound)
# ---------------------------------------------------------------------------

def _make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model       = LLM_MODEL,
        temperature = 0,
        api_key     = os.environ["OPENAI_API_KEY"],
    ).bind_tools(TOOLS)


def agent_node(state: AgentState) -> dict:
    """
    Call the LLM with the full message history.
    Returns either an AIMessage with tool_calls (→ tools node)
    or a final AIMessage with no tool_calls (→ END).
    """
    llm = _make_llm()

    # Prepend the system prompt on the first turn only (it is not stored in
    # state to keep the history clean for display in the UI).
    from langchain_core.messages import SystemMessage
    messages_with_system = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + state["messages"]

    log.info("agent_node_called", n_messages=len(state["messages"]))
    response: AIMessage = llm.invoke(messages_with_system)

    # Collect any NCT IDs mentioned in the response text
    new_sources = _NCT_PATTERN.findall(response.content or "")

    return {
        "messages": [response],
        "sources":  list(dict.fromkeys(state.get("sources", []) + new_sources)),
    }


# ---------------------------------------------------------------------------
# Node: tools (execute whichever tool(s) the LLM requested)
# ---------------------------------------------------------------------------

# Map tool name → callable so we can dispatch by name
_TOOL_MAP: dict = {t.name: t for t in TOOLS}


def tools_node(state: AgentState) -> dict:
    """
    Execute every tool call in the last AIMessage and return ToolMessages.
    Collects any NCT IDs found in tool outputs.
    """
    last_message: AIMessage = state["messages"][-1]
    tool_messages = []
    new_sources: list[str] = []

    for tool_call in last_message.tool_calls:
        name   = tool_call["name"]
        args   = tool_call["args"]
        tc_id  = tool_call["id"]

        log.info("tool_called", tool=name, args=args)

        if name not in _TOOL_MAP:
            result = f"Error: unknown tool '{name}'"
        else:
            try:
                result = _TOOL_MAP[name].invoke(args)
            except Exception as exc:
                result = f"Error running {name}: {exc}"

        new_sources += _NCT_PATTERN.findall(str(result))
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tc_id)
        )

    return {
        "messages": tool_messages,
        "sources":  list(dict.fromkeys(state.get("sources", []) + new_sources)),
    }


# ---------------------------------------------------------------------------
# Routing: should we run tools or are we done?
# ---------------------------------------------------------------------------

def _should_continue(state: AgentState) -> str:
    """Return 'tools' if the LLM issued tool calls, else 'end'."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Compile and return the ClinicalMind LangGraph agent."""
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)

    graph.add_edge(START, "agent")

    graph.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", "end": END},
    )

    graph.add_edge("tools", "agent")   # loop back after tool execution

    return graph.compile()


# Singleton — imported by other modules
agent_graph = build_graph()


# ---------------------------------------------------------------------------
# Convenience wrapper (used by the Streamlit UI and tests)
# ---------------------------------------------------------------------------

@traceable(name="clinicalmind_agent", run_type="chain")
def run_agent(question: str) -> dict:
    """
    Run the agent on a single question.

    Returns a dict with:
      - answer   : the agent's final response text
      - sources  : list of NCT IDs cited during the run
      - messages : full message history for display
      - error    : error string if something went wrong, else None
    """
    if not question.strip():
        raise ValueError("Question must not be empty.")

    initial_state: AgentState = {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "sources":  [],
    }

    t0 = time.monotonic()
    try:
        final_state = agent_graph.invoke(initial_state)
    except Exception as exc:
        log.error("agent_run_failed", error=str(exc),
                  response_time_ms=round((time.monotonic() - t0) * 1000))
        return {
            "answer":   "I encountered an error. Please try again.",
            "sources":  [],
            "messages": [HumanMessage(content=question)],
            "error":    str(exc),
        }

    # The final answer is the content of the last AIMessage
    last_ai = next(
        (m for m in reversed(final_state["messages"]) if isinstance(m, AIMessage)),
        None,
    )
    answer = last_ai.content if last_ai else "No answer generated."

    log.info(
        "agent_run_complete",
        question=question[:80],
        sources=len(final_state["sources"]),
        response_time_ms=round((time.monotonic() - t0) * 1000),
    )

    return {
        "answer":   answer,
        "sources":  final_state["sources"],
        "messages": final_state["messages"],
        "error":    None,
    }


# ---------------------------------------------------------------------------
# Entry point — smoke test with stub tools
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    questions = [
        "What are common exclusion criteria for Phase 3 diabetes trials?",
        "How many recruiting hypertension trials are there?",
    ]
    for q in questions:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        result = run_agent(q)
        print(f"\nA: {result['answer']}")
        print(f"Sources: {result['sources']}")
        if result["error"]:
            print(f"[Error: {result['error']}]")
