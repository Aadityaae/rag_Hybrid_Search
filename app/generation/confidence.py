"""Composite confidence scoring + graceful low-confidence fallback.

Combines three signals:
  - retrieval_confidence: how relevant the top retrieved chunks were
  - citation_coverage: % of claims with a verified citation
  - answer_completeness: LLM-judge estimate of whether the answer
    addressed all parts of the question

If the composite score is below `min_retrieval_confidence`, the caller
should prefer the structured "don't know" response over the raw answer.
"""
from __future__ import annotations

import json

from openai import OpenAI, OpenAIError

from app.config import settings
from app.llm_client import get_chat_client, rate_limited_chat_completion
from app.openai_utils import openai_retry, raise_clear_error

COMPLETENESS_PROMPT = """Question: {question}

Answer: {answer}

On a 0-10 scale, how completely does the answer address ALL parts of the question \
(not whether it's correct, just whether it's complete)? Respond with ONLY a JSON object: \
{{"completeness": <int 0-10>}}"""


@openai_retry
def _completeness_score(client: OpenAI, question: str, answer: str) -> float:
    resp = rate_limited_chat_completion(
        client,
        model=settings.llm_judge_model,
        messages=[{"role": "user", "content": COMPLETENESS_PROMPT.format(question=question, answer=answer)}],
        temperature=0,
        max_tokens=20,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    return float(data.get("completeness", 0)) / 10.0


def retrieval_confidence(final_chunks: list[dict]) -> float:
    """Uses rerank_score (0-10, LLM judge) when available, else falls back
    to normalized fusion/vector score. Weighted toward the top result."""
    if not final_chunks:
        return 0.0
    scores = []
    for c in final_chunks:
        if "rerank_score" in c:
            scores.append(c["rerank_score"] / 10.0)
        elif "score" in c:
            scores.append(max(0.0, min(1.0, c["score"])))
        else:
            scores.append(0.5)
    # weight the top chunk heaviest -- it's what generation leans on most
    weights = [1.0 / (i + 1) for i in range(len(scores))]
    return sum(s * w for s, w in zip(scores, weights)) / sum(weights)


def score_answer(question: str, answer_text: str, final_chunks: list[dict],
                  citation_report: dict, client: OpenAI = None) -> dict:
    client = client or get_chat_client()
    r_conf = retrieval_confidence(final_chunks)
    coverage = citation_report["citation_coverage"]
    try:
        completeness = _completeness_score(client, question, answer_text)
    except OpenAIError as e:
        raise_clear_error(e)

    composite = 0.4 * r_conf + 0.35 * coverage + 0.25 * completeness

    return {
        "retrieval_confidence": round(r_conf, 3),
        "citation_coverage": round(coverage, 3),
        "answer_completeness": round(completeness, 3),
        "composite_confidence": round(composite, 3),
        "below_threshold": composite < settings.min_retrieval_confidence,
    }


def build_dont_know_response(question: str, final_chunks: list[dict]) -> dict:
    """Structured fallback when confidence is too low to trust the answer:
    says what was found, what wasn't, and which docs to check manually."""
    found_sources = sorted({
        c["metadata"].get("source_file", "unknown") for c in final_chunks
    }) if final_chunks else []

    return {
        "answer": (
            "I don't have enough confidently-grounded information in the indexed "
            "documentation to answer this fully. Answering anyway would risk making "
            "something up, so I'm flagging it instead."
        ),
        "what_was_found": (
            f"The closest matches came from: {', '.join(found_sources)}."
            if found_sources else "No sufficiently relevant documents were retrieved."
        ),
        "what_was_missing": "No chunk cleared the confidence threshold for a grounded answer to this question.",
        "suggested_documents_to_check": found_sources,
        "is_fallback": True,
    }
