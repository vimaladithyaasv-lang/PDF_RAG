"""
eval_rag.py — Evaluation harness for the RAG pipeline.

Measures two things that "does it run" tests can't:

1. Retrieval quality — given a labeled set of (query, expected_source_file)
   pairs, what fraction of the time does the top-k retrieval actually surface
   a chunk from the expected file? (recall@k)

2. Answer quality — using the LLM itself as a judge (a common, pragmatic
   approach when no human-labeled "gold answer" dataset exists), score each
   generated answer on:
     - faithfulness: is the answer actually supported by the retrieved
       context, or does it hallucinate beyond it?
     - relevance: does the answer actually address the question asked?

This is intentionally lightweight (no external eval framework dependency) so
it can run anywhere RAGEngine runs, including CI. It is NOT a substitute for
human review on a production system with real user queries; treat scores as
a *signal to investigate*, not a certification.

Usage:
    python eval_rag.py --testset eval_testset.json

Or import and call programmatically:
    from eval_rag import run_evaluation
    report = run_evaluation(engine, testset)
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from rag_engine import RAGEngine, GROQ_MODEL


@dataclass
class EvalCase:
    query: str
    expected_source: Optional[str] = None  # filename the answer should draw from
    top_k: int = 5


@dataclass
class EvalResult:
    query: str
    expected_source: Optional[str]
    retrieved_sources: List[str]
    retrieval_hit: Optional[bool]   # None if expected_source wasn't specified
    answer: str
    faithfulness_score: Optional[float]  # 0-1, None if judging failed
    faithfulness_reason: str
    relevance_score: Optional[float]
    relevance_reason: str
    latency_seconds: float


@dataclass
class EvalReport:
    results: List[EvalResult] = field(default_factory=list)

    @property
    def recall_at_k(self) -> Optional[float]:
        labeled = [r for r in self.results if r.retrieval_hit is not None]
        if not labeled:
            return None
        return sum(1 for r in labeled if r.retrieval_hit) / len(labeled)

    @property
    def mean_faithfulness(self) -> Optional[float]:
        scored = [r.faithfulness_score for r in self.results if r.faithfulness_score is not None]
        return sum(scored) / len(scored) if scored else None

    @property
    def mean_relevance(self) -> Optional[float]:
        scored = [r.relevance_score for r in self.results if r.relevance_score is not None]
        return sum(scored) / len(scored) if scored else None

    @property
    def mean_latency(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_seconds for r in self.results) / len(self.results)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "n_cases": len(self.results),
                "recall_at_k": self.recall_at_k,
                "mean_faithfulness": self.mean_faithfulness,
                "mean_relevance": self.mean_relevance,
                "mean_latency_seconds": round(self.mean_latency, 2),
            },
            "cases": [
                {
                    "query": r.query,
                    "expected_source": r.expected_source,
                    "retrieved_sources": r.retrieved_sources,
                    "retrieval_hit": r.retrieval_hit,
                    "answer": r.answer,
                    "faithfulness_score": r.faithfulness_score,
                    "faithfulness_reason": r.faithfulness_reason,
                    "relevance_score": r.relevance_score,
                    "relevance_reason": r.relevance_reason,
                    "latency_seconds": round(r.latency_seconds, 2),
                }
                for r in self.results
            ],
        }


def _judge_with_llm(
    engine: RAGEngine, question: str, context: str, answer: str
) -> Dict[str, object]:
    """
    Ask the LLM to score its own (or another run's) answer against the
    retrieved context. Returns a dict with faithfulness/relevance scores
    in [0, 1] and short reasons. Falls back to None scores if the judge
    call fails or returns unparseable output — callers should treat that
    as "couldn't evaluate," not "scored zero."
    """
    judge_prompt = (
        "You are evaluating the quality of an AI assistant's answer to a "
        "question, given the context it was allowed to use.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT PROVIDED TO THE ASSISTANT:\n{context}\n\n"
        f"ASSISTANT'S ANSWER:\n{answer}\n\n"
        "Score two things on a 0.0-1.0 scale:\n"
        "1. faithfulness: Is every claim in the answer actually supported by "
        "the context? 1.0 = fully grounded, 0.0 = mostly fabricated/unsupported.\n"
        "2. relevance: Does the answer actually address the question asked? "
        "1.0 = directly answers it, 0.0 = off-topic or non-answer.\n\n"
        "Respond ONLY with a JSON object, no other text, no markdown fences:\n"
        '{"faithfulness": <float>, "faithfulness_reason": "<one sentence>", '
        '"relevance": <float>, "relevance_reason": "<one sentence>"}'
    )
    try:
        response = engine.groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        return {
            "faithfulness_score": float(parsed.get("faithfulness")),
            "faithfulness_reason": str(parsed.get("faithfulness_reason", "")),
            "relevance_score": float(parsed.get("relevance")),
            "relevance_reason": str(parsed.get("relevance_reason", "")),
        }
    except Exception as e:
        return {
            "faithfulness_score": None,
            "faithfulness_reason": f"Judge call failed: {e}",
            "relevance_score": None,
            "relevance_reason": f"Judge call failed: {e}",
        }


def run_evaluation(
    engine: RAGEngine,
    cases: List[EvalCase],
    judge: bool = True,
) -> EvalReport:
    """
    Run every case through the engine and score retrieval + answer quality.

    judge=False skips the LLM-as-judge calls (faster, no extra API cost) and
    only computes retrieval recall@k — useful for quick CI checks.
    """
    report = EvalReport()

    for case in cases:
        start = time.monotonic()

        retrieved = engine.retrieve(case.query, top_k=case.top_k, min_score=None)
        retrieved_sources = [fname for _, fname, _, _, _ in retrieved]

        retrieval_hit = None
        if case.expected_source is not None:
            retrieval_hit = case.expected_source in retrieved_sources

        answer, used_chunks = engine.answer(case.query, top_k=case.top_k)
        latency = time.monotonic() - start

        faithfulness_score = faithfulness_reason = None
        relevance_score = relevance_reason = None
        if judge and used_chunks:
            context = "\n\n".join(c[0] for c in used_chunks)
            judged = _judge_with_llm(engine, case.query, context, answer)
            faithfulness_score  = judged["faithfulness_score"]
            faithfulness_reason = judged["faithfulness_reason"]
            relevance_score      = judged["relevance_score"]
            relevance_reason     = judged["relevance_reason"]

        report.results.append(
            EvalResult(
                query=case.query,
                expected_source=case.expected_source,
                retrieved_sources=retrieved_sources,
                retrieval_hit=retrieval_hit,
                answer=answer,
                faithfulness_score=faithfulness_score,
                faithfulness_reason=faithfulness_reason or "",
                relevance_score=relevance_score,
                relevance_reason=relevance_reason or "",
                latency_seconds=latency,
            )
        )

    return report


def load_testset(path: str) -> List[EvalCase]:
    """
    Load a JSON test set: a list of {"query": ..., "expected_source": ...}
    objects. expected_source is optional (omit it to evaluate answer quality
    only, without checking retrieval recall).
    """
    with open(path) as f:
        raw = json.load(f)
    return [
        EvalCase(
            query=item["query"],
            expected_source=item.get("expected_source"),
            top_k=item.get("top_k", 5),
        )
        for item in raw
    ]


def print_report(report: EvalReport) -> None:
    d = report.to_dict()
    s = d["summary"]
    print("\n=== RAG Evaluation Report ===")
    print(f"Cases run:           {s['n_cases']}")
    if s["recall_at_k"] is not None:
        print(f"Retrieval recall@k:  {s['recall_at_k']:.1%}")
    if s["mean_faithfulness"] is not None:
        print(f"Mean faithfulness:   {s['mean_faithfulness']:.2f} / 1.00")
    if s["mean_relevance"] is not None:
        print(f"Mean relevance:      {s['mean_relevance']:.2f} / 1.00")
    print(f"Mean latency:        {s['mean_latency_seconds']:.2f}s")
    print()
    for c in d["cases"]:
        flag = ""
        if c["retrieval_hit"] is False:
            flag += " ⚠ retrieval miss"
        if c["faithfulness_score"] is not None and c["faithfulness_score"] < 0.5:
            flag += " ⚠ low faithfulness"
        print(f"- {c['query'][:60]!r}{flag}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--testset", required=True, help="Path to JSON test set.")
    parser.add_argument("--index-dir", default=".rag_index", help="Saved index directory to load.")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-judge scoring (retrieval-only).")
    parser.add_argument("--out", default=None, help="Optional path to write full JSON report.")
    args = parser.parse_args()

    engine = RAGEngine()  # reads GROQ_API_KEY from env
    if not engine.load_index(args.index_dir):
        print(f"No saved index found at '{args.index_dir}'. Index some documents first.", file=sys.stderr)
        sys.exit(1)

    cases = load_testset(args.testset)
    report = run_evaluation(engine, cases, judge=not args.no_judge)
    print_report(report)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
