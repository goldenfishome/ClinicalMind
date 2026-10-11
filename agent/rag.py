"""
Basic RAG Q&A: retrieval → prompt → GPT-4o → grounded answer.

This is the core question-answering function. The LangGraph agent
(Feature 12) wraps this as its RAG Search Tool.

LangSmith tracing is enabled automatically when LANGCHAIN_TRACING_V2=true
and LANGCHAIN_API_KEY are set in .env. Every call to ask() appears as a
named run in the LangSmith dashboard showing inputs, outputs, latency,
and token usage.

Usage:
    python3 -m agent.rag
"""

import os
import re

import structlog
from dotenv import load_dotenv
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI, APIError

from agent.logging_config import setup_logging
from retrieval.search import search
from agent.prompts import (
    RAG_SYSTEM_PROMPT,
    RAG_CONTEXT_TEMPLATE,
    build_rag_context,
)

load_dotenv()
setup_logging()
log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LLM_MODEL   = "gpt-4o"
DEFAULT_TOPK = 5

# NCT IDs follow the pattern NCT + 8 digits
_NCT_PATTERN = re.compile(r"NCT\d{8}")

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _extract_nct_ids(text: str) -> list[str]:
    """Return unique NCT IDs mentioned in the answer text, in order of appearance."""
    return list(dict.fromkeys(_NCT_PATTERN.findall(text)))


@traceable(name="rag_ask", run_type="chain")
def ask(
    question: str,
    top_k: int = DEFAULT_TOPK,
    condition: str | None = None,
    phase: str | None = None,
    status: str | None = None,
) -> dict:
    """
    Answer a clinical trial question using RAG.

    Steps:
      1. Retrieve top-k relevant chunks from pgvector (with optional filters)
      2. Build a grounded prompt from those chunks
      3. Call GPT-4o for synthesis
      4. Extract cited NCT IDs from the answer

    Returns a dict with:
      - answer       : the GPT-4o response string
      - sources      : list of NCT IDs cited in the answer
      - chunks_used  : the raw retrieved chunks (for UI display and RAGAS eval)
      - error        : set if the LLM call failed, answer will be a safe message
    """
    if not question.strip():
        raise ValueError("Question must not be empty.")

    # Step 1 — retrieve
    chunks = search(question, top_k=top_k, condition=condition, phase=phase, status=status)

    if not chunks:
        log.warning("no_chunks_retrieved", question=question[:80])
        return {
            "answer":      "I could not find any relevant trial data for your question.",
            "sources":     [],
            "chunks_used": [],
            "error":       None,
        }

    # Step 2 — build prompt
    context      = build_rag_context(chunks)
    user_message = RAG_CONTEXT_TEMPLATE.format(context=context, question=question)

    # Step 3 — call GPT-4o
    # wrap_openai adds a child span in LangSmith showing prompt, response,
    # tokens, and latency. Falls back to plain OpenAI when tracing is off.
    _tracing_on = os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    _raw_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    client = wrap_openai(_raw_client) if _tracing_on else _raw_client
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system",  "content": RAG_SYSTEM_PROMPT},
                {"role": "user",    "content": user_message},
            ],
            temperature=0,      # deterministic — grounded answers only
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()

    except APIError as exc:
        log.error("llm_call_failed", error=str(exc))
        return {
            "answer":      "I encountered an error reaching the language model. Please try again.",
            "sources":     [],
            "chunks_used": chunks,
            "error":       str(exc),
        }

    # Step 4 — extract cited NCT IDs
    sources = _extract_nct_ids(answer)

    log.info(
        "rag_answer_generated",
        question=question[:80],
        chunks_used=len(chunks),
        sources_cited=len(sources),
    )

    return {
        "answer":      answer,
        "sources":     sources,
        "chunks_used": chunks,
        "error":       None,
    }


# ---------------------------------------------------------------------------
# Entry point — smoke test against a live database
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    questions = [
        "What are common exclusion criteria for Phase 3 diabetes trials?",
        "What interventions are used in hypertension trials?",
    ]

    for q in questions:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        result = ask(q, top_k=5)
        print(f"\nA: {result['answer']}")
        print(f"\nSources: {result['sources']}")
        if result["error"]:
            print(f"[Error: {result['error']}]")
