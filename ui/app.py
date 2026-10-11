"""
ClinicalMind — Streamlit UI

A chat interface for the LangGraph ReAct agent. Each user message is sent
to run_agent(), which chooses between RAG Search, SQL Lookup, and Web Search
to construct a grounded, cited answer.

Run:
    streamlit run ui/app.py
"""

import re
import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — allow imports from project root when launched via streamlit
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.logging_config import setup_logging          # noqa: E402
from agent.graph import run_agent                       # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

setup_logging()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title = "ClinicalMind",
    page_icon  = "🧬",
    layout     = "wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NCT_PATTERN = re.compile(r"NCT\d{8}")
NCT_URL     = "https://clinicaltrials.gov/study/{}"

EXAMPLE_QUESTIONS = [
    "What are the exclusion criteria for Phase 3 diabetes trials?",
    "How many recruiting hypertension trials are in the database?",
    "What interventions are used in cardiac arrest trials?",
    "Which sponsors have the most cardiac arrest trials?",
    "What were the most important FDA drug approvals in 2025?",
    "What blood glucose levels are required for diabetes trial eligibility?",
]

CONDITION_OPTIONS = ["", "Diabetes", "Hypertension", "Cardiac Arrest"]
PHASE_OPTIONS     = ["", "PHASE1", "PHASE2", "PHASE3", "PHASE4"]
STATUS_OPTIONS    = ["", "RECRUITING", "COMPLETED", "ACTIVE_NOT_RECRUITING"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linkify_nct_ids(text: str) -> str:
    """Replace NCT IDs in text with markdown hyperlinks to ClinicalTrials.gov."""
    return NCT_PATTERN.sub(
        lambda m: f"[{m.group()}]({NCT_URL.format(m.group())})",
        text,
    )


def _tool_label(name: str) -> str:
    icons = {"rag_search": "🔬", "sql_lookup": "🗄️", "web_search": "🌐"}
    labels = {
        "rag_search":  "RAG Search",
        "sql_lookup":  "SQL Lookup",
        "web_search":  "Web Search",
    }
    icon  = icons.get(name, "🔧")
    label = labels.get(name, name)
    return f"{icon} {label}"


def _extract_tool_calls(messages: list) -> list[dict]:
    """
    Walk the message list and pair each AIMessage tool_call with its
    corresponding ToolMessage result.
    Returns a list of {name, args, result} dicts.
    """
    calls   = []
    tm_map  = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                calls.append({
                    "name":   tc["name"],
                    "args":   tc["args"],
                    "result": tm_map.get(tc["id"], "(no result)"),
                })
    return calls


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    # Each entry: {"role": "user"|"assistant", "content": str,
    #              "tool_calls": list, "sources": list}
    st.session_state.history = []

if "prefill" not in st.session_state:
    st.session_state.prefill = ""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🧬 ClinicalMind")
    st.caption("Clinical trial intelligence powered by GPT-4o + pgvector")

    st.divider()

    st.subheader("💡 Example questions")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, key=f"ex_{q[:30]}", use_container_width=True):
            st.session_state.prefill = q

    st.divider()

    st.subheader("🔧 RAG filters (optional)")
    st.caption("Pre-filter the trial database before semantic search.")

    condition_filter = st.selectbox("Condition", CONDITION_OPTIONS)
    phase_filter     = st.selectbox("Phase",     PHASE_OPTIONS)
    status_filter    = st.selectbox("Status",    STATUS_OPTIONS)

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.prefill = ""
        st.rerun()

    st.divider()
    st.caption(
        "Data: ClinicalTrials.gov · Conditions: Diabetes, Hypertension, "
        "Cardiac Arrest · ~115K chunks"
    )


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.header("ClinicalMind", divider="gray")
st.caption(
    "Ask anything about clinical trial eligibility, design, outcomes, or sponsors. "
    "The agent chooses between **RAG Search** 🔬, **SQL Lookup** 🗄️, and **Web Search** 🌐 automatically."
)

# Render conversation history
for entry in st.session_state.history:
    with st.chat_message(entry["role"], avatar="🧑" if entry["role"] == "user" else "🧬"):
        if entry["role"] == "assistant":
            st.markdown(_linkify_nct_ids(entry["content"]))

            # Tool call details — collapsed by default
            if entry.get("tool_calls"):
                with st.expander(
                    f"🔍 Tool calls ({len(entry['tool_calls'])} used)",
                    expanded=False,
                ):
                    for tc in entry["tool_calls"]:
                        st.markdown(f"**{_tool_label(tc['name'])}**")
                        if tc["args"]:
                            args_display = "  \n".join(
                                f"- `{k}`: {v}" for k, v in tc["args"].items() if v
                            )
                            st.markdown(args_display)
                        with st.expander("Result", expanded=False):
                            st.text(tc["result"][:1000])
                        st.divider()

            # Cited NCT IDs as badge-style links
            if entry.get("sources"):
                st.markdown(
                    "**Cited trials:** "
                    + " · ".join(
                        f"[{s}]({NCT_URL.format(s)})"
                        for s in entry["sources"]
                    )
                )
        else:
            st.markdown(entry["content"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

# Pre-fill from sidebar example button, then clear so it doesn't re-trigger
default_input = st.session_state.prefill
if default_input:
    st.session_state.prefill = ""

user_input = st.chat_input(
    "Ask about clinical trials…",
    key="chat_input",
)

# Accept either typed input or a sidebar example button click
question = user_input or default_input

if question:
    # Show the user bubble immediately
    st.session_state.history.append({
        "role":       "user",
        "content":    question,
        "tool_calls": [],
        "sources":    [],
    })
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    # Build filter kwargs — only pass non-empty selections
    # (the agent forwards these to rag_search if it decides to use that tool)
    extra_context = {}
    if condition_filter:
        extra_context["condition"] = condition_filter
    if phase_filter:
        extra_context["phase"] = phase_filter
    if status_filter:
        extra_context["status"] = status_filter

    # Append filter context to question text so the agent can pass them on
    augmented_question = question
    if extra_context:
        filter_str = ", ".join(f"{k}={v}" for k, v in extra_context.items())
        augmented_question = f"{question} [filters: {filter_str}]"

    # Run agent with spinner
    with st.chat_message("assistant", avatar="🧬"):
        with st.spinner("Thinking…"):
            result = run_agent(augmented_question)

        answer     = result["answer"]
        sources    = result["sources"]
        tool_calls = _extract_tool_calls(result["messages"])

        # Render answer with clickable NCT IDs
        st.markdown(_linkify_nct_ids(answer))

        # Tool call details
        if tool_calls:
            with st.expander(
                f"🔍 Tool calls ({len(tool_calls)} used)",
                expanded=False,
            ):
                for tc in tool_calls:
                    st.markdown(f"**{_tool_label(tc['name'])}**")
                    if tc["args"]:
                        args_display = "  \n".join(
                            f"- `{k}`: {v}" for k, v in tc["args"].items() if v
                        )
                        st.markdown(args_display)
                    with st.expander("Result", expanded=False):
                        st.text(tc["result"][:1000])
                    st.divider()

        # Cited NCT ID links
        if sources:
            st.markdown(
                "**Cited trials:** "
                + " · ".join(
                    f"[{s}]({NCT_URL.format(s)})"
                    for s in sources
                )
            )

    # Persist to session history
    st.session_state.history.append({
        "role":       "assistant",
        "content":    answer,
        "tool_calls": tool_calls,
        "sources":    sources,
    })
