"""Generates case_study.md from saved eval result JSON files (Phase 6,
step 2 of the original build plan: "lead with the numbers, explain why
hybrid beats dense-only, show the chunking comparison").

Run after you have eval results saved (run_eval.py auto-saves to
data/eval_results/ by default):

    python scripts/generate_case_study.py \\
        --full-suite data/eval_results/full_suite_<ts>.json \\
        --retrieval-modes data/eval_results/compare_retrieval_modes_<ts>.json \\
        --chunking data/eval_results/compare_chunking_<ts>.json

Any of the three inputs can be omitted -- the report just notes that
section wasn't run yet, rather than failing.
"""
import argparse
import json
from pathlib import Path


def load(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"Warning: {p} not found, skipping that section.")
        return None
    return json.loads(p.read_text())


def render_full_suite(data: dict | None) -> str:
    if not data:
        return "*(Full eval suite not yet run -- run `python -m app.eval.run_eval` and pass " \
               "`--full-suite` to this script.)*\n"

    s = data["summary"]
    by_type = s["by_type"]
    lines = [
        f"Across **{s['n_cases']} questions** spanning lookup, multi-hop, no-answer, and "
        f"ambiguous types, the pipeline achieves:\n",
        f"| Metric | Score |",
        f"|---|---|",
        f"| Answer correctness (LLM-judged vs. golden answer) | **{s['avg_correctness']:.1%}** |",
        f"| Faithfulness (claims grounded in retrieved context) | **{s['avg_faithfulness']:.1%}** |",
        f"| Citation accuracy (cited claims that hold up under verification) | **{s['avg_citation_accuracy']:.1%}** |",
        f"| Retrieval relevance (right source docs retrieved) | **{s['avg_retrieval_relevance']:.1%}** |",
        f"| Fallback rate (declined to answer) | {s['fallback_rate']:.1%} |",
        "",
        "**By question type:**\n",
        "| Type | n | Avg. correctness |",
        "|---|---|---|",
    ]
    for t, v in sorted(by_type.items(), key=lambda kv: -kv[1]["avg_correctness"]):
        lines.append(f"| {t} | {v['n']} | {v['avg_correctness']:.1%} |")

    lines.append("")
    worst_type = min(by_type.items(), key=lambda kv: kv[1]["avg_correctness"])
    best_type = max(by_type.items(), key=lambda kv: kv[1]["avg_correctness"])
    lines.append(
        f"The weakest category is **{worst_type[0]}** ({worst_type[1]['avg_correctness']:.1%} vs. "
        f"{best_type[1]['avg_correctness']:.1%} for {best_type[0]}). This is the eval framework "
        f"doing its job: it surfaces a real, specific weakness rather than hiding behind a single "
        f"blended average."
    )
    return "\n".join(lines)


def render_retrieval_modes(data: dict | None) -> str:
    if not data:
        return "*(Hybrid vs. dense-only comparison not yet run -- run `python -m app.eval.run_eval " \
               "--compare-retrieval-modes` and pass `--retrieval-modes` to this script. This " \
               "comparison makes no LLM calls, so it costs nothing to run.)*\n"

    s = data["summary"]
    dense = s["avg_dense_only_relevance"]
    hybrid = s["avg_hybrid_relevance"]
    delta = hybrid - dense if (dense is not None and hybrid is not None) else None

    lines = [
        f"Measured on {s['n_cases']} questions with known source documents "
        f"(retrieval_relevance = fraction of the correct source docs actually retrieved):\n",
        f"| Mode | Avg. retrieval relevance |",
        f"|---|---|",
        f"| Dense-only | {dense:.1%} |" if dense is not None else "| Dense-only | n/a |",
        f"| Hybrid (dense + BM25 + RRF) | {hybrid:.1%} |" if hybrid is not None else "| Hybrid | n/a |",
    ]
    if delta is not None:
        if delta > 0:
            lines.append(f"\nHybrid retrieval outperforms dense-only by **{delta:+.1%}** on this corpus. "
                          f"This matters most for technical documentation with exact identifiers "
                          f"(config keys, error codes, CLI flags like `nw-cli deploy rollback`) that "
                          f"embedding similarity alone can miss but BM25 keyword matching catches directly.")
        elif delta == 0:
            lines.append("\nBoth modes performed equally on this run -- on a small, topically "
                          "clustered corpus like this synthetic one, dense retrieval alone can "
                          "already find the right docs. The gap tends to widen on larger, noisier "
                          "corpora with more exact-match terms (error codes, config keys, IDs).")
        else:
            lines.append(f"\nDense-only slightly outperformed hybrid by {abs(delta):.1%} on this run -- "
                          f"worth checking the RRF dense/sparse weighting (`fusion_dense_weight` / "
                          f"`fusion_sparse_weight` in app/config.py) rather than assuming hybrid is "
                          f"always better; it depends on the corpus.")
    return "\n".join(lines)


def render_chunking(data: dict | None) -> str:
    if not data:
        return "*(Chunking strategy comparison not yet run -- run `python -m app.eval.run_eval " \
               "--compare-chunking --sample-per-type 2` for a cheap, representative comparison, " \
               "then pass `--chunking` to this script.)*\n"

    lines = ["| Strategy | n | Avg. correctness | Avg. faithfulness | Avg. citation accuracy |",
             "|---|---|---|---|---|"]
    for strategy, result in data.items():
        s = result["summary"]
        lines.append(
            f"| {strategy} | {s['n_cases']} | {s['avg_correctness']:.1%} | "
            f"{s['avg_faithfulness']:.1%} | {s['avg_citation_accuracy']:.1%} |"
        )
    best = max(data.items(), key=lambda kv: kv[1]["summary"]["avg_correctness"] or 0)
    lines.append(f"\n**{best[0]}** chunking scored highest on this sample "
                 f"({best[1]['summary']['avg_correctness']:.1%} correctness).")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-suite", type=str, default=None)
    parser.add_argument("--retrieval-modes", type=str, default=None)
    parser.add_argument("--chunking", type=str, default=None)
    parser.add_argument("--output", type=str, default="case_study.md")
    parser.add_argument("--corpus-description", type=str,
                         default="a document corpus (see data/raw/ for the indexed source files)",
                         help="One-line description of what was indexed, e.g. "
                              "'a synthetic internal engineering-docs corpus (8 files)'")
    args = parser.parse_args()

    full_suite = load(args.full_suite)
    retrieval_modes = load(args.retrieval_modes)
    chunking = load(args.chunking)
    n_questions = full_suite["summary"]["n_cases"] if full_suite else "N"

    doc = f"""# RAG Pipeline with Hybrid Search Over Internal Docs -- Case Study

## Summary

A production-shaped Retrieval-Augmented Generation system built over {args.corpus_description}.
Combines dense (embedding) and sparse (BM25) retrieval via Reciprocal Rank Fusion, reranks with
an LLM-as-judge, generates grounded answers with inline citations, and verifies every citation
against its source before reporting a confidence score.

Runs entirely on free infrastructure: local `sentence-transformers` embeddings and Groq's
free-tier Llama models for generation and judging -- no paid API required.

## Results

{render_full_suite(full_suite)}

## Hybrid vs. Dense-Only Retrieval

{render_retrieval_modes(retrieval_modes)}

## Chunking Strategy Comparison

{render_chunking(chunking)}

## Known limitations

- Eval set is a {n_questions}-question starter, not the 50+ a fuller case study would use --
  extend `app/eval/golden_qa.json` before publishing these numbers widely.
- Enumeration-style questions ("list every X across the document") score lower than simple
  lookups when the corpus has more matching instances than fit in the top-k retrieved/reranked
  chunks -- a structural limit of top-k retrieval, not a generation bug. Worth calling out
  explicitly rather than averaging away.
- The smaller/faster judge model (Llama 3.1 8B, chosen to fit the free-tier rate limit) is
  less reliable at following "don't cite while declining to answer" instructions than a
  larger model -- see the no_answer category's faithfulness score for where this shows up.
"""

    Path(args.output).write_text(doc)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
