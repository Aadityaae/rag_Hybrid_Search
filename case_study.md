# RAG Pipeline with Hybrid Search Over Internal Docs -- Case Study

## Summary

A production-shaped Retrieval-Augmented Generation system built over a single 54-section internal reference document (a stress-test corpus with heavy near-duplicate content -- each section covers one (topic, country) pair, e.g. "Tactical Analysis -- England", with largely repeated paragraph structure across sections).
Combines dense (embedding) and sparse (BM25) retrieval via Reciprocal Rank Fusion, reranks with
an LLM-as-judge, generates grounded answers with inline citations, and verifies every citation
against its source before reporting a confidence score.

Runs entirely on free infrastructure: local `sentence-transformers` embeddings and Groq's
free-tier Llama models for generation and judging -- no paid API required.

## Results

Across **20 questions** spanning lookup, multi-hop, no-answer, and ambiguous types, the pipeline achieves:

| Metric | Score |
|---|---|
| Answer correctness (LLM-judged vs. golden answer) | **75.5%** |
| Faithfulness (claims grounded in retrieved context) | **60.0%** |
| Citation accuracy (cited claims that hold up under verification) | **75.0%** |
| Retrieval relevance (right source docs retrieved) | **100.0%** |
| Fallback rate (declined to answer) | 0.0% |

**By question type:**

| Type | n | Avg. correctness |
|---|---|---|
| no_answer | 4 | 90.0% |
| lookup | 10 | 83.0% |
| multi_hop | 4 | 55.0% |
| ambiguous | 2 | 50.0% |

The weakest category is **ambiguous** (50.0% vs. 90.0% for no_answer). This is the eval framework doing its job: it surfaces a real, specific weakness rather than hiding behind a single blended average.

## Hybrid vs. Dense-Only Retrieval

*(Hybrid vs. dense-only comparison not yet run -- run `python -m app.eval.run_eval --compare-retrieval-modes` and pass `--retrieval-modes` to this script. This comparison makes no LLM calls, so it costs nothing to run.)*


## Chunking Strategy Comparison

*(Chunking strategy comparison not yet run -- run `python -m app.eval.run_eval --compare-chunking --sample-per-type 2` for a cheap, representative comparison, then pass `--chunking` to this script.)*


## Known limitations

- Eval set is a 20-question starter, not the 50+ a fuller case study would use --
  extend `app/eval/golden_qa.json` before publishing these numbers widely.
- Enumeration-style questions ("list every X across the document") score lower than simple
  lookups when the corpus has more matching instances than fit in the top-k retrieved/reranked
  chunks -- a structural limit of top-k retrieval, not a generation bug. Worth calling out
  explicitly rather than averaging away.
- The smaller/faster judge model (Llama 3.1 8B, chosen to fit the free-tier rate limit) is
  less reliable at following "don't cite while declining to answer" instructions than a
  larger model -- see the no_answer category's faithfulness score for where this shows up.
