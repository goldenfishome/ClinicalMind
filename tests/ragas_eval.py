"""
RAGAS evaluation for ClinicalMind RAG pipeline.

Runs 20 hand-labeled questions through the RAG pipeline and scores:
  - Faithfulness      : Does the answer stay grounded in the retrieved context?
  - Answer Relevancy  : Is the answer relevant to the question asked?
  - Context Precision : Are the retrieved chunks actually relevant to the question?

Scores range from 0.0 to 1.0 — higher is better.

Usage:
    python3 -m tests.ragas_eval              # full eval, all 20 questions
    python3 -m tests.ragas_eval --n 5        # quick smoke test, first 5 questions
"""

import argparse
import json
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness
from ragas.run_config import RunConfig

from agent.rag import ask

load_dotenv()

# Suppress deprecation warnings for the legacy metric singletons —
# they are deprecated in favour of collections but are the only
# approach that passes isinstance(m, Metric) in RAGAS 0.4.x.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
LLM_MODEL    = "gpt-4o"
EMBED_MODEL  = "text-embedding-3-small"

# RAGAS makes several sequential LLM calls per sample (claim extraction,
# NLI scoring, etc.).  Key tuning choices:
#   timeout=600   — context_precision loops over 3 chunks sequentially, each
#                   taking ~80-120 s; give 10 min total per job
#   max_retries=2 — avoid the default 10 retries compounding timeouts
#   max_wait=10   — short backoff so a transient failure retries quickly
#   max_workers=2 — two parallel samples; reduces API rate-limit contention
RAGAS_RUN_CONFIG = RunConfig(timeout=600, max_retries=2, max_wait=10, max_workers=2)


# ---------------------------------------------------------------------------
# Build evaluation dataset by running each question through the RAG pipeline
# ---------------------------------------------------------------------------

def build_ragas_dataset(questions: list[dict]) -> EvaluationDataset:
    """
    Run each question through ask() and collect the inputs/outputs
    into the format RAGAS 0.4.x expects (EvaluationDataset of SingleTurnSamples).
    """
    samples = []

    for i, item in enumerate(questions, 1):
        q  = item["question"]
        gt = item["ground_truth"]

        print(f"  [{i}/{len(questions)}] {q[:70]}...")
        # top_k=3 keeps context_precision's per-chunk LLM calls within the
        # RAGAS timeout (5 calls at ~40 s each → 200 s; 3 calls → ~120 s).
        result = ask(q, top_k=3)

        samples.append(SingleTurnSample(
            user_input         = q,
            response           = result["answer"],
            retrieved_contexts = [c["chunk_text"] for c in result["chunks_used"]],
            reference          = gt,
        ))

    return EvaluationDataset(samples=samples)


# ---------------------------------------------------------------------------
# Run RAGAS scoring
# ---------------------------------------------------------------------------

def run_eval(n: int | None = None) -> dict:
    """
    Load the eval dataset, run questions through the pipeline, score with RAGAS.
    Returns a dict of metric_name -> score.
    """
    questions = json.loads(DATASET_PATH.read_text())
    if n:
        questions = questions[:n]

    print(f"\nRunning {len(questions)} questions through the RAG pipeline...")
    dataset = build_ragas_dataset(questions)

    # LangChain-wrapped LLM and embeddings — accepted by ragas.evaluate()
    # in addition to its own BaseRagasLLM / BaseRagasEmbeddings types.
    llm = ChatOpenAI(
        model       = LLM_MODEL,
        temperature = 0,
        api_key     = os.environ["OPENAI_API_KEY"],
    )
    emb = OpenAIEmbeddings(
        model   = EMBED_MODEL,
        api_key = os.environ["OPENAI_API_KEY"],
    )

    # Run each metric independently to avoid parallel LLM saturation.
    # Running all three at once fires ~15 simultaneous API calls per sample
    # (faithfulness alone decomposes answers into claims + NLI per chunk),
    # which trips rate limits and causes TimeoutErrors.  Three sequential
    # passes are slower in wall time but each pass stays well within limits.
    def _eval_one(metric, needs_embeddings: bool = False) -> float:
        """Score a single metric over the full dataset; return nanmean or 0.0."""
        import math
        print(f"  → {metric.name}...")
        kwargs = dict(
            dataset          = dataset,
            metrics          = [metric],
            llm              = llm,
            run_config       = RAGAS_RUN_CONFIG,
            raise_exceptions = False,
            show_progress    = True,
        )
        if needs_embeddings:
            kwargs["embeddings"] = emb
        r = evaluate(**kwargs)
        try:
            v = r._repr_dict.get(metric.name, float("nan"))
            return 0.0 if math.isnan(float(v)) else round(float(v), 3)
        except (TypeError, ValueError):
            return 0.0

    import time
    print("\nScoring with RAGAS (three sequential passes — may take ~15 min)...")
    f_score  = _eval_one(faithfulness)
    time.sleep(5)   # brief pause between passes to ease API rate limits
    ar_score = _eval_one(answer_relevancy, needs_embeddings=True)
    time.sleep(5)
    cp_score = _eval_one(context_precision)

    scores = {
        "faithfulness":      f_score,
        "answer_relevancy":  ar_score,
        "context_precision": cp_score,
    }

    return scores


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None,
                        help="Number of questions to evaluate (default: all 20)")
    args = parser.parse_args()

    scores = run_eval(n=args.n)

    print("\n" + "="*50)
    print("RAGAS EVALUATION RESULTS")
    print("="*50)
    for metric, score in scores.items():
        bar   = "█" * int(score * 20)
        grade = "✅" if score >= 0.7 else "⚠️ " if score >= 0.5 else "❌"
        print(f"  {metric:<22} {score:.3f}  {bar:<20} {grade}")
    print("="*50)
    print("\nScore guide:  ✅ ≥ 0.70   ⚠️  ≥ 0.50   ❌ < 0.50")
    print("\nFull results saved to: tests/ragas_results.json")

    results_path = Path(__file__).parent / "ragas_results.json"
    results_path.write_text(json.dumps({"n_questions": args.n or 20, "scores": scores}, indent=2))
