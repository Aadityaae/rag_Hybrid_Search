"""Second-pass reranker over the top-N fused candidates. Uses an
LLM-as-judge (cheap model) to score each chunk's relevance to the actual
question on a 0-10 scale, then keeps the top `rerank_top_n`. This catches
cases where RRF surfaces a chunk that merely shares vocabulary/embeddings
with the query but doesn't actually answer it.
"""
from __future__ import annotations

import json

from openai import OpenAI, OpenAIError

from app.config import settings
from app.llm_client import get_chat_client, rate_limited_chat_completion
from app.openai_utils import openai_retry, raise_clear_error

RERANK_PROMPT = """You score how relevant a document excerpt is to a question, 0-10.
10 = directly and completely answers the question.
5 = topically related but doesn't answer it.
0 = unrelated.

Question: {question}

Excerpt:
\"\"\"{excerpt}\"\"\"

Respond with ONLY a JSON object: {{"score": <int 0-10>}}"""


@openai_retry
def _score_one(client: OpenAI, question: str, excerpt: str) -> int:
    resp = rate_limited_chat_completion(
        client,
        model=settings.llm_judge_model,
        messages=[{"role": "user", "content": RERANK_PROMPT.format(question=question, excerpt=excerpt[:2000])}],
        temperature=0,
        max_tokens=20,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    return int(data.get("score", 0))


def rerank(question: str, candidates: list[dict], top_n: int = None, client: OpenAI = None) -> list[dict]:
    top_n = top_n or settings.rerank_top_n
    if not candidates:
        return []
    client = client or get_chat_client()
    scored = []
    try:
        for c in candidates:
            score = _score_one(client, question, c["text"])
            item = dict(c)
            item["rerank_score"] = score
            scored.append(item)
    except OpenAIError as e:
        raise_clear_error(e)
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_n]
