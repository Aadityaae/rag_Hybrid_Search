"""Phase 4: automated eval metrics, computed per test case:

  answer_correctness  - LLM-as-judge vs. golden answer (0-1)
  faithfulness        - are all claims grounded in retrieved context? (0-1,
                         reuses the citation verification report)
  retrieval_relevance - were the right source docs retrieved at all? (0-1,
                         checks golden source_docs against retrieved chunk
                         source files -- doesn't require an LLM call)
  citation_accuracy   - do citations actually support the claims they're
                         attached to? (0-1, from the citation report)

Run the full suite with `python -m app.eval.run_eval`.
"""
from __future__ import annotations

import json

from openai import OpenAI, OpenAIError

from app.config import settings
from app.llm_client import get_chat_client, rate_limited_chat_completion
from app.openai_utils import openai_retry, raise_clear_error

CORRECTNESS_PROMPT = """Compare the MODEL ANSWER to the GOLDEN (reference) ANSWER for the given question.
Score 0-10 on whether the model answer conveys the same key facts as the golden answer (wording
can differ). If the golden answer says the question is unanswerable from the docs and the model
answer also correctly declines/flags low confidence, that counts as fully correct (10).

QUESTION: {question}
GOLDEN ANSWER: {golden}
MODEL ANSWER: {model_answer}

Respond with ONLY a JSON object: {{"score": <int 0-10>}}"""


@openai_retry
def answer_correctness(question: str, golden_answer: str, model_answer: str, client: OpenAI = None) -> float:
    client = client or get_chat_client()
    try:
        resp = rate_limited_chat_completion(
            client,
            model=settings.llm_judge_model,
            messages=[{"role": "user", "content": CORRECTNESS_PROMPT.format(
                question=question, golden=golden_answer, model_answer=model_answer)}],
            temperature=0,
            max_tokens=20,
            response_format={"type": "json_object"},
        )
    except OpenAIError as e:
        raise_clear_error(e)
    data = json.loads(resp.choices[0].message.content)
    return float(data.get("score", 0)) / 10.0


def faithfulness(citation_report: dict) -> float:
    """Fraction of all claims (cited or not) that are verified as grounded.
    Uncited factual claims count against faithfulness too."""
    total = citation_report["total_claims"]
    if total == 0:
        return 1.0
    return citation_report["verified_claims"] / total


def citation_accuracy(citation_report: dict) -> float:
    """Of the claims that WERE cited, what fraction hold up under verification."""
    cited = citation_report["cited_claims"]
    if cited == 0:
        return None  # not applicable -- no citations were made
    return citation_report["verified_claims"] / cited


def retrieval_relevance(expected_source_docs: list[str], final_chunks: list[dict]) -> float:
    """Fraction of expected source documents that appear among the
    retrieved chunks' source files. For no_answer cases (empty expected
    list), score 1.0 if nothing from the corpus was forced in confidently
    -- handled by the caller checking the fallback path instead."""
    if not expected_source_docs:
        return None  # not applicable, handled separately for no_answer cases
    retrieved_files = {c["metadata"].get("source_file") for c in final_chunks}
    hits = sum(1 for d in expected_source_docs if d in retrieved_files)
    return hits / len(expected_source_docs)
