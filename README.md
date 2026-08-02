# Internal Docs RAG — Hybrid Search with Grounded Citations

A production-shaped Retrieval-Augmented Generation system: multi-format ingestion,
hybrid (dense + sparse) retrieval with reciprocal rank fusion and reranking,
grounded generation with citation verification and confidence scoring, an eval
framework, and a FastAPI + Streamlit deployment.

## Cost

Runs entirely on free tiers:

- **Embeddings**: local `sentence-transformers` model (`all-MiniLM-L6-v2`) — runs on
  your own CPU, no API key, no per-call cost, no rate limits. Weights download once
  (~90MB) from Hugging Face on first use.
- **Generation + judging** (answer generation, reranking, citation verification,
  eval): [Groq](https://console.groq.com/keys), which has a genuinely free tier for
  open models like Llama 3.3, no credit card required.

To swap in a different provider (OpenAI, another OpenAI-compatible endpoint, a local
Ollama server), only `app/llm_client.py` (chat) and `app/ingestion/embeddings.py`
(embeddings) need to change — nothing else in the pipeline depends on the provider.

## Architecture

```
Documents (.md/.txt/.html/.pdf)
        │
        ▼
  Loaders (app/ingestion/loaders.py) ── normalize to plaintext + metadata
        │
        ▼
  Chunking (app/ingestion/chunking.py) ── fixed | recursive | semantic
        │
        ▼
  Embeddings + Dedup (app/ingestion/embeddings.py, dedup.py)
        │
        ├──────────────► ChromaDB (dense vectors)  ─┐
        └──────────────► BM25 index (sparse)        ├─► HybridRetriever
                                                       │   (app/retrieval/engine.py)
                                                       ▼
                                     RRF Fusion → Reranker (LLM-as-judge) → top-k chunks
                                                       │
                                                       ▼
                          Grounded Generation (app/generation/prompt.py)
                                                       │
                          Citation Verification (citation_verify.py)
                                                       │
                          Confidence Scoring (confidence.py) → answer or "I don't know"
                                                       │
                                                       ▼
                              FastAPI (/v1/ask, /v1/documents, /v1/ingest)
                                                       │
                                                       ▼
                                         Streamlit dashboard (frontend/app.py)
```

## Quickstart

### 1. Local (no Docker)

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY (free at console.groq.com/keys)

# Generates the synthetic sample corpus (data/raw) and indexes it
python scripts/seed.py --strategy recursive

# Run the API
uvicorn app.api.main:app --reload

# In another terminal, run the dashboard
streamlit run frontend/app.py
```

Then open http://localhost:8501, or hit the API directly:

```bash
curl -X POST localhost:8000/v1/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "How does rollback work?"}'
```

### 2. Docker

```bash
cp .env.example .env   # fill in GROQ_API_KEY (free at console.groq.com/keys)
docker-compose up --build
```

This seeds the sample corpus automatically, then starts the API on `:8000` and the
dashboard on `:8501`.

## Using your own documents

Drop `.md`, `.txt`, `.html`, or `.pdf` files into `data/raw/` (replacing or alongside
the synthetic sample docs) and re-run:

```bash
python scripts/seed.py --strategy recursive
```

or hit `POST /v1/ingest` with `{"directory": "data/raw", "strategy": "recursive"}`.

## Chunking strategies

Three switchable strategies live in `app/ingestion/chunking.py`:

- **fixed** — fixed-size character windows with overlap (baseline)
- **recursive** — structure-aware splitting on headings → paragraphs → sentences
- **semantic** — groups sentences until embedding similarity drops (topic boundary)

## Evaluation

A 22-question golden dataset (`app/eval/golden_qa.json`) covers lookup, multi-hop,
no-answer, and ambiguous question types. **This is a starter set** — for a portfolio
case study, expand it to 50+ pairs against your real corpus.

```bash
# Eval the currently-indexed collection
python -m app.eval.run_eval

# Re-ingest with each chunking strategy and compare metrics side by side
python -m app.eval.run_eval --compare-chunking
```

Metrics computed per question: answer correctness (LLM-as-judge vs. golden answer),
faithfulness (fraction of claims grounded in context), citation accuracy (fraction of
*cited* claims that hold up under verification), and retrieval relevance (did the
right source docs get retrieved).

> The pipeline self-throttles to stay under Groq's free-tier RPM cap (`GROQ_RPM_LIMIT`
> in `.env`, default 25/min per model). Each question can make 5-15+ judge calls
> (rerank + per-claim citation checks + completeness), so the full 22-question eval
> suite will visibly pause to pace itself rather than fail with a 429 -- that's
> expected, not a hang.

## Tests

```bash
pip install pytest
pytest tests/ -v
```

Covers loaders, all three chunking strategies, dedup, BM25, and RRF fusion — the
parts of the pipeline that don't require API calls, so they run without a key.

## Notes on scope / what to extend before treating this as final

- The golden eval set is a 22-question starter, not the full 50+ the case study should
  cite — extend it against your real corpus before publishing numbers.
- Semantic chunking is intentionally embedding-driven and injected via `embed_fn`, so
  swapping the embedding provider doesn't touch the chunker itself.
- `min_retrieval_confidence` and the RRF dense/sparse weighting (`app/config.py`) are
  starting points — tune both against your eval suite, not by feel.
- The reranker and citation verifier both call an LLM judge per-chunk/per-claim, which
  adds latency and cost proportional to `rerank_candidate_pool` and answer length —
  worth capping or batching before scaling traffic.
