import numpy as np
import pytest

from app.ingestion.chunking import Chunk, chunk_fixed, chunk_recursive, chunk_semantic
from app.ingestion.dedup import DuplicateFilter
from app.ingestion.loaders import Document, load_markdown, load_html, _clean_text
from app.retrieval.bm25_store import BM25Store
from app.retrieval.fusion import reciprocal_rank_fusion


def test_clean_text_collapses_whitespace():
    assert _clean_text("a   b\n\n\n\nc") == "a b\n\nc"


def test_load_markdown_splits_on_headings(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Title\nintro text\n## Section A\nbody a\n## Section B\nbody b\n")
    docs = load_markdown(p)
    headings = [d.section_heading for d in docs]
    assert "Title" in headings and "Section A" in headings and "Section B" in headings


def test_load_html_extracts_headings_and_paragraphs(tmp_path):
    p = tmp_path / "doc.html"
    p.write_text("<h1>Intro</h1><p>hello world</p><h2>Details</h2><p>more text</p>")
    docs = load_html(p)
    assert any(d.section_heading == "Intro" for d in docs)
    assert any("hello world" in d.text for d in docs)


def test_chunk_fixed_respects_overlap():
    doc = Document(doc_id="d1", source_file="d1.txt", text="x" * 1000)
    chunks = chunk_fixed(doc, chunk_size=200, overlap=50)
    assert all(c.chunking_strategy == "fixed" for c in chunks)
    assert len(chunks) > 1
    # consecutive chunks should overlap in source offsets -> total chars > raw text length
    assert sum(len(c.text) for c in chunks) > len(doc.text)


def test_chunk_recursive_stays_under_reasonable_bound():
    doc = Document(doc_id="d1", source_file="d1.md", text="Sentence one. " * 200)
    chunks = chunk_recursive(doc, chunk_size=300, overlap=30)
    assert all(len(c.text) <= 400 for c in chunks)  # some slack for separator boundaries
    assert all(c.chunking_strategy == "recursive" for c in chunks)


def test_chunk_semantic_uses_injected_embed_fn():
    doc = Document(doc_id="d1", source_file="d1.txt",
                    text="Cats are mammals. Cats like to sleep. Rockets use liquid fuel. Rockets are fast.")

    def fake_embed(sentences):
        # crude fake: "cat" sentences cluster near [1,0], "rocket" sentences near [0,1]
        vecs = []
        for s in sentences:
            vecs.append([1.0, 0.0] if "at" in s.lower() else [0.0, 1.0])
        return np.array(vecs)

    chunks = chunk_semantic(doc, embed_fn=fake_embed, similarity_drop_threshold=0.5)
    assert len(chunks) >= 2
    assert all(c.chunking_strategy == "semantic" for c in chunks)


def test_dedup_filter_drops_near_duplicates():
    dedup = DuplicateFilter(threshold=0.95)
    base = np.array([1.0, 0.0, 0.0])
    near_dup = np.array([0.99, 0.01, 0.0])
    different = np.array([0.0, 1.0, 0.0])

    is_dup1, _ = dedup.is_duplicate("a", base)
    is_dup2, sim2 = dedup.is_duplicate("b", near_dup)
    is_dup3, _ = dedup.is_duplicate("c", different)

    assert is_dup1 is False
    assert is_dup2 is True and sim2 > 0.95
    assert is_dup3 is False


def test_bm25_store_finds_keyword_matches():
    store = BM25Store()
    store.build(
        chunk_ids=["1", "2", "3"],
        texts=["the canary deploy failed with high error rates",
               "onboarding covers accounts and setup",
               "feature flags let you roll out gradually"],
        metadatas=[{}, {}, {}],
    )
    results = store.query("canary error rates", top_k=2)
    assert results[0]["chunk_id"] == "1"


def test_rrf_prioritizes_items_ranked_in_both_lists():
    dense = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
    sparse = [{"chunk_id": "c"}, {"chunk_id": "d"}]
    fused = reciprocal_rank_fusion(dense, sparse)
    assert fused[0]["chunk_id"] == "c"
    assert set(fused[0]["matched_by"]) == {"dense", "sparse"}


def test_insufficient_quota_fails_fast_with_clear_message():
    """Regression test: an insufficient_quota RateLimitError must surface a
    clear OpenAIConfigError immediately, not get retried and buried inside
    a tenacity.RetryError. This is the shared policy used by every chat-
    completion call site (reranker, citation verification, confidence
    scoring, generation, eval) -- tested once here at the source."""
    import httpx
    from openai import OpenAIError, RateLimitError

    from app.openai_utils import OpenAIConfigError, openai_retry, raise_clear_error

    resp = httpx.Response(
        429,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        json={"error": {"message": "You exceeded your current quota", "code": "insufficient_quota"}},
    )
    err = RateLimitError("quota exceeded", response=resp, body=resp.json())

    call_count = {"n": 0}

    @openai_retry
    def flaky_call():
        call_count["n"] += 1
        raise err

    with pytest.raises(RateLimitError):
        flaky_call()
    # Must fail on the first attempt, not retry 3-4 times, since this error
    # code never recovers on its own.
    assert call_count["n"] == 1

    with pytest.raises(OpenAIConfigError, match="insufficient_quota"):
        try:
            flaky_call()
        except OpenAIError as e:
            raise_clear_error(e)


def test_transient_rate_limit_is_retried():
    """A rate limit WITHOUT insufficient_quota (i.e. still has room, just
    hit the requests-per-minute cap) should be retried, not failed fast."""
    import httpx
    from openai import RateLimitError

    from app.openai_utils import openai_retry

    resp = httpx.Response(
        429,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        json={"error": {"message": "Rate limit reached, try again shortly", "code": "rate_limit_exceeded"}},
    )
    err = RateLimitError("rate limited", response=resp, body=resp.json())

    call_count = {"n": 0}

    @openai_retry
    def flaky_call():
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise err
        return "ok"

    assert flaky_call() == "ok"
    assert call_count["n"] == 2


def test_rate_limiter_throttles_bursts_but_allows_steady_rate():
    """Uses a tiny window (0.4s, 2 calls) so the test runs fast: the 3rd
    call within the window must block until the window has room, proving
    the pipeline won't burst past a provider's RPM cap."""
    import time

    from app.rate_limiter import RateLimiter

    limiter = RateLimiter(max_calls=2, period_seconds=0.4)

    start = time.monotonic()
    limiter.acquire()  # call 1 -- immediate
    limiter.acquire()  # call 2 -- immediate
    limiter.acquire()  # call 3 -- must wait for call 1 to fall out of the window
    elapsed = time.monotonic() - start

    assert elapsed >= 0.35  # allow small scheduling slack below the 0.4s window
