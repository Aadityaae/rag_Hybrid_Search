# Demo Walkthrough Script (target: under 4 minutes)

Record your screen with the API running (`python -m uvicorn app.api.main:app --reload`) and
the Swagger UI or Streamlit dashboard open. Suggested flow:

## 0:00-0:30 -- Ingestion
- Show `data/raw/` with the 8 source docs (or your own docs, if swapped in).
- Run `python scripts/seed.py --strategy recursive` (or show it already run) and point out
  the summary output: documents loaded, chunks embedded, chunks deduped.
- One line of narration: "Multi-format loader, structure-aware chunking, local embeddings,
  indexed into both a vector store and a BM25 keyword index that stay in sync."

## 0:30-1:30 -- Ask a normal question
- In the Streamlit dashboard (or Swagger `/v1/ask`), ask a clean lookup question, e.g.
  "What is the rate limit for read-only API keys?"
- Point out on screen, in order:
  1. The answer with inline `[1]` citations
  2. The confidence breakdown (retrieval confidence / citation coverage / completeness)
  3. Expand "Retrieved chunks" -- show the actual source text the answer is grounded in
  4. Expand "Citation verification report" -- show each claim checked against its source

## 1:30-2:15 -- Multi-hop question
- Ask something that needs two docs, e.g. "What tool pages the on-call engineer, and what
  monitoring stack feeds alerts into it?"
- Point out the retrieved chunks come from *two different source files* -- this is hybrid
  retrieval pulling relevant context that a single-document search would miss.

## 2:15-2:50 -- Citation verification catching a hallucination
- This is the strongest moment in the demo -- show a *real* case where the citation
  verifier flagged something. If you don't have one on hand, this is worth deliberately
  reproducing: ask a judge-adjacent question and show the `citation_report` with a claim
  marked `"status": "unsupported"` or `"uncited"`.
- Narration: "The model's own citations aren't trusted blindly -- every claim gets checked
  against the actual source text by a second LLM pass before the confidence score is computed."

## 2:50-3:30 -- Hybrid vs. dense-only toggle
- In the Streamlit dashboard, enable "Compare hybrid vs. dense-only" and ask a question with
  an exact identifier in it (a config key, CLI command, or error code -- these are where BM25
  keyword matching earns its keep over embeddings alone).
- Show the two retrieved-chunk lists side by side; narrate the difference if there is one.

## 3:30-4:00 -- Close with the numbers
- Cut to `case_study.md` (or read the numbers off it): correctness, faithfulness, citation
  accuracy across the golden eval set, and the chunking-strategy comparison if you've run it.
- One line: "All of this runs on free infrastructure -- local embeddings, Groq's free tier for
  generation and judging -- so the whole thing costs nothing to run or demo."

---

## Recording notes
- Use OBS Studio, Windows Game Bar (Win+G), or Loom -- all free, all sufficient for a
  screen-capture demo like this.
- Do a dry run of the questions first so you already know which ones give clean answers vs.
  which trigger the fallback path -- both are worth showing, but know which is which going in.
- Trim dead air from the rate-limit pauses in post, or narrate over them ("this pause is the
  pipeline self-throttling to stay under the free tier's rate limit -- a real production
  system would either pay for higher limits or batch these calls differently").
