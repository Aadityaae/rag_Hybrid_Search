"""Runs the full eval suite against the live pipeline and, optionally,
across all three chunking strategies to produce a comparison report
(Phase 4, step 3). Requires GROQ_API_KEY and a populated index.

Prints progress per question as it runs -- each question makes several
throttled judge calls, so with the free-tier rate limit the full 22-case
suite can take a while. Use --limit for a quick smoke test, or
--sample-per-type to run a smaller *representative* set (one or more of
each question type) for a cheaper --compare-chunking run.

Usage:
    python -m app.eval.run_eval                          # full 22-case eval
    python -m app.eval.run_eval --limit 3                 # quick smoke test
    python -m app.eval.run_eval --case-id q16 --verbose   # inspect one case in full
    python -m app.eval.run_eval --compare-chunking \\
        --sample-per-type 2                                # cheap, representative comparison
    python -m app.eval.run_eval --output data/eval_results/run1.json  # save results to disk
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

from app.config import GOLDEN_DATASET_PATH, RAW_DIR, ROOT_DIR
from app.eval.metrics import answer_correctness, citation_accuracy, faithfulness, retrieval_relevance
from app.generation.service import RagService
from app.ingestion.pipeline import ingest_directory
from app.llm_client import get_chat_client
from app.retrieval.bm25_store import BM25Store
from app.retrieval.engine import HybridRetriever
from app.retrieval.vector_store import VectorStore

EVAL_RESULTS_DIR = ROOT_DIR / "data" / "eval_results"


def load_golden(limit: int = None, case_id: str = None, sample_per_type: int = None) -> list[dict]:
    with open(GOLDEN_DATASET_PATH) as f:
        cases = json.load(f)

    if case_id:
        matches = [c for c in cases if c["id"] == case_id]
        if not matches:
            raise SystemExit(f"No golden case with id={case_id!r}")
        return matches

    if sample_per_type:
        by_type: dict[str, list[dict]] = defaultdict(list)
        for c in cases:
            by_type[c["type"]].append(c)
        sampled = []
        for t, group in by_type.items():
            sampled.extend(group[:sample_per_type])
        return sampled

    return cases[:limit] if limit else cases


def run_case(service: RagService, case: dict, client: OpenAI, verbose: bool = False) -> dict:
    result = service.ask(case["question"])
    final_chunks = result["retrieval"]["final_chunks"]

    metrics = {"id": case["id"], "type": case["type"], "question": case["question"]}

    if result.get("is_fallback"):
        metrics["correctness"] = 1.0 if case["type"] == "no_answer" else 0.0
        metrics["faithfulness"] = None
        metrics["citation_accuracy"] = None
        metrics["retrieval_relevance"] = retrieval_relevance(case["source_docs"], final_chunks)
        metrics["fallback_triggered"] = True
    else:
        metrics["correctness"] = answer_correctness(case["question"], case["expected_answer"], result["answer"], client=client)
        metrics["faithfulness"] = faithfulness(result["citation_report"])
        metrics["citation_accuracy"] = citation_accuracy(result["citation_report"])
        metrics["retrieval_relevance"] = retrieval_relevance(case["source_docs"], final_chunks)
        metrics["fallback_triggered"] = False

    if verbose:
        metrics["answer"] = result.get("answer")
        metrics["citation_report"] = result.get("citation_report")
        metrics["confidence"] = result.get("confidence")
        metrics["retrieved_sources"] = [
            c["metadata"].get("source_file") for c in final_chunks
        ]

    return metrics


def run_cases_with_progress(service: RagService, cases: list[dict], client: OpenAI, verbose: bool = False) -> list[dict]:
    results = []
    for i, case in enumerate(cases, start=1):
        t0 = time.monotonic()
        print(f"[{i}/{len(cases)}] {case['id']} ({case['type']}): {case['question'][:70]!r} ...", flush=True)
        metrics = run_case(service, case, client, verbose=verbose)
        elapsed = time.monotonic() - t0
        print(f"    -> correctness={metrics['correctness']} fallback={metrics['fallback_triggered']} ({elapsed:.1f}s)", flush=True)
        if verbose:
            print(f"    answer: {metrics['answer']}")
            print(f"    citation_report: {json.dumps(metrics['citation_report'], indent=2)}")
            print(f"    confidence: {json.dumps(metrics['confidence'], indent=2)}")
        results.append(metrics)
    return results


def summarize(results: list[dict]) -> dict:
    def avg(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return round(statistics.mean(vals), 3) if vals else None

    by_type = {}
    for t in {r["type"] for r in results}:
        subset = [r for r in results if r["type"] == t]
        subset_correctness = [r["correctness"] for r in subset if r.get("correctness") is not None]
        by_type[t] = {
            "n": len(subset),
            "avg_correctness": round(statistics.mean(subset_correctness), 3) if subset_correctness else None,
        }

    return {
        "n_cases": len(results),
        "avg_correctness": avg("correctness"),
        "avg_faithfulness": avg("faithfulness"),
        "avg_citation_accuracy": avg("citation_accuracy"),
        "avg_retrieval_relevance": avg("retrieval_relevance"),
        "fallback_rate": round(sum(1 for r in results if r["fallback_triggered"]) / len(results), 3),
        "by_type": by_type,
    }


def compare_hybrid_vs_dense(limit: int = None, case_id: str = None, sample_per_type: int = None) -> dict:
    """Compares hybrid (dense+sparse+RRF) vs. dense-only retrieval using
    retrieval_relevance against the golden source_docs. Deliberately skips
    both generation AND the reranker, so this makes ZERO LLM calls --
    just local embeddings + BM25 -- and costs nothing to run."""
    vs = VectorStore()
    bm25 = BM25Store()
    bm25.load()
    retriever = HybridRetriever(vector_store=vs, bm25_store=bm25)

    cases = load_golden(limit=limit, case_id=case_id, sample_per_type=sample_per_type)
    # Only cases with known source docs are meaningful here (no_answer cases
    # have an empty source_docs list by design and would trivially score 0/0).
    cases = [c for c in cases if c["source_docs"]]

    rows = []
    for c in cases:
        dense = retriever.retrieve(c["question"], use_reranker=False, apply_hybrid=False)
        hybrid = retriever.retrieve(c["question"], use_reranker=False, apply_hybrid=True)
        rows.append({
            "id": c["id"],
            "type": c["type"],
            "dense_only_relevance": retrieval_relevance(c["source_docs"], dense["final_chunks"]),
            "hybrid_relevance": retrieval_relevance(c["source_docs"], hybrid["final_chunks"]),
        })

    def avg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(statistics.mean(vals), 3) if vals else None

    return {
        "rows": rows,
        "summary": {
            "n_cases": len(rows),
            "avg_dense_only_relevance": avg("dense_only_relevance"),
            "avg_hybrid_relevance": avg("hybrid_relevance"),
        },
    }


def run_full_suite(collection_name: str = None, limit: int = None, case_id: str = None,
                    sample_per_type: int = None, verbose: bool = False) -> dict:
    vs = VectorStore(collection_name=collection_name) if collection_name else VectorStore()
    bm25 = BM25Store()
    bm25.load()
    retriever = HybridRetriever(vector_store=vs, bm25_store=bm25)
    client = get_chat_client()
    service = RagService(retriever=retriever, client=client)

    cases = load_golden(limit=limit, case_id=case_id, sample_per_type=sample_per_type)
    print(f"Running {len(cases)} eval case(s)...", flush=True)
    results = run_cases_with_progress(service, cases, client, verbose=verbose)
    return {"results": results, "summary": summarize(results)}


def compare_chunking_strategies(limit: int = None, sample_per_type: int = None, verbose: bool = False) -> dict:
    report = {}
    for strategy in ["fixed", "recursive", "semantic"]:
        print(f"\n=== strategy: {strategy} ===", flush=True)
        collection = f"eval_{strategy}"
        vs = VectorStore(collection_name=collection)
        vs.reset()
        bm25 = BM25Store()
        ingest_directory(RAW_DIR, strategy=strategy, vector_store=vs, bm25_store=bm25)
        retriever = HybridRetriever(vector_store=vs, bm25_store=bm25)
        client = get_chat_client()
        service = RagService(retriever=retriever, client=client)
        cases = load_golden(limit=limit, sample_per_type=sample_per_type)
        print(f"Running {len(cases)} eval case(s)...", flush=True)
        results = run_cases_with_progress(service, cases, client, verbose=verbose)
        report[strategy] = {"results": results, "summary": summarize(results)}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-chunking", action="store_true")
    parser.add_argument("--compare-retrieval-modes", action="store_true",
                         help="Free: compares hybrid vs dense-only retrieval relevance (no LLM calls)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only run the first N golden questions (quick smoke test)")
    parser.add_argument("--case-id", type=str, default=None,
                         help="Run only one specific golden question by id (e.g. q16), for debugging")
    parser.add_argument("--sample-per-type", type=int, default=None,
                         help="Run only the first N questions of EACH type (lookup/multi_hop/no_answer/"
                              "ambiguous) -- a cheap, representative subset for --compare-chunking")
    parser.add_argument("--verbose", action="store_true",
                         help="Print full answer text, citation report, and confidence per case")
    parser.add_argument("--output", type=str, default=None,
                         help="Save results as JSON to this path (default: auto-named under data/eval_results/)")
    args = parser.parse_args()

    if args.compare_chunking:
        out = compare_chunking_strategies(limit=args.limit, sample_per_type=args.sample_per_type, verbose=args.verbose)
    elif args.compare_retrieval_modes:
        out = compare_hybrid_vs_dense(limit=args.limit, case_id=args.case_id, sample_per_type=args.sample_per_type)
    else:
        out = run_full_suite(limit=args.limit, case_id=args.case_id,
                              sample_per_type=args.sample_per_type, verbose=args.verbose)

    print("\n" + "=" * 60)
    print(json.dumps(out, indent=2))

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        tag = "compare_chunking" if args.compare_chunking else (
            "compare_retrieval_modes" if args.compare_retrieval_modes else (args.case_id or "full_suite"))
        out_path = EVAL_RESULTS_DIR / f"{tag}_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved results to {out_path}")
