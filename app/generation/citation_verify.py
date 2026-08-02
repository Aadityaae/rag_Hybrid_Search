"""Citation verification -- the quality layer most RAG systems skip.

Splits the generated answer into claim sentences, pairs each with its
cited context block(s), and asks an LLM judge whether the block actually
supports the claim. Unsupported citations are flagged so the confidence
scorer (and the UI) can surface them instead of silently trusting the
model's own citations.
"""
from __future__ import annotations

import json
import re

from openai import OpenAI, OpenAIError

from app.config import settings
from app.llm_client import get_chat_client, rate_limited_chat_completion
from app.openai_utils import openai_retry, raise_clear_error

VERIFY_PROMPT = """Does the SOURCE TEXT support the CLAIM? A source supports a claim if the \
claim's factual content can be verified from the source, even if worded differently. \
It does NOT need to be an exact match.

CLAIM: {claim}

SOURCE TEXT:
\"\"\"{source}\"\"\"

Respond with ONLY a JSON object: {{"supported": true|false, "reason": "<one short sentence>"}}"""


def split_claims(answer_text: str) -> list[str]:
    """Splits the answer into sentence-level claims, keeping trailing
    citation markers attached to the sentence they belong to."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", answer_text.strip())
    return [s.strip() for s in sentences if s.strip()]


def extract_citations(claim: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", claim)})


@openai_retry
def _verify_one(client: OpenAI, claim: str, source: str) -> dict:
    resp = rate_limited_chat_completion(
        client,
        model=settings.llm_judge_model,
        messages=[{"role": "user", "content": VERIFY_PROMPT.format(claim=claim, source=source[:2500])}],
        temperature=0,
        max_tokens=100,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def verify_citations(answer_text: str, context_chunks: list[dict], client: OpenAI = None) -> dict:
    client = client or get_chat_client()
    claims = split_claims(answer_text)

    results = []
    try:
        for claim in claims:
            cited = extract_citations(claim)
            if not cited:
                # Uncited factual sentence -- flag distinctly from "verified false".
                results.append({"claim": claim, "cited_indices": [], "status": "uncited"})
                continue
            claim_verifications = []
            for idx in cited:
                if not (1 <= idx <= len(context_chunks)):
                    claim_verifications.append({"index": idx, "supported": False, "reason": "citation index out of range"})
                    continue
                source_text = context_chunks[idx - 1]["text"]
                verdict = _verify_one(client, claim, source_text)
                claim_verifications.append({"index": idx, **verdict})
            supported = any(v.get("supported") for v in claim_verifications)
            results.append({
                "claim": claim,
                "cited_indices": cited,
                "status": "verified" if supported else "unsupported",
                "verifications": claim_verifications,
            })
    except OpenAIError as e:
        raise_clear_error(e)

    total_cited_claims = sum(1 for r in results if r["cited_indices"])
    verified_claims = sum(1 for r in results if r["status"] == "verified")
    uncited_claims = sum(1 for r in results if r["status"] == "uncited")

    coverage = (verified_claims / len(results)) if results else 0.0

    return {
        "claims": results,
        "total_claims": len(results),
        "cited_claims": total_cited_claims,
        "verified_claims": verified_claims,
        "uncited_claims": uncited_claims,
        "unsupported_claims": total_cited_claims - verified_claims,
        "citation_coverage": coverage,
    }
