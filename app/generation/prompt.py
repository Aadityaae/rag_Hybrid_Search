"""Grounded generation. Builds numbered context blocks from retrieved
chunks, instructs the model to answer only from that context and cite
with bracketed references, and to explicitly say when context is
insufficient rather than hallucinate.
"""
from __future__ import annotations

import re

from openai import OpenAI, OpenAIError

from app.config import settings
from app.llm_client import get_chat_client, rate_limited_chat_completion
from app.openai_utils import raise_clear_error

SYSTEM_PROMPT = """You are an internal documentation assistant. Answer the user's question \
using ONLY the numbered context blocks provided below. Follow these rules strictly:

1. Every factual claim must be followed by a bracketed citation like [1] or [2][3] \
referencing the context block(s) that support it.
2. Do not use any knowledge outside the provided context.
3. If the context does not contain enough information to answer (fully or partially), \
say so explicitly. Do not guess or fabricate an answer.
4. If you are declining to answer because the context is insufficient, do NOT include \
any bracketed citations in that sentence -- citations mean "this specific fact is backed \
by this source," and a decline-to-answer is not a factual claim about the source content.
5. Be concise and direct. Do not repeat the question.
"""


def build_context_blocks(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        meta = c.get("metadata", {})
        loc = meta.get("source_file", "unknown")
        if meta.get("section_heading"):
            loc += f" — {meta['section_heading']}"
        elif meta.get("page_number"):
            loc += f" — p.{meta['page_number']}"
        blocks.append(f"[{i}] (source: {loc})\n{c['text']}")
    return "\n\n".join(blocks)


def generate_answer(question: str, chunks: list[dict], client: OpenAI = None) -> dict:
    client = client or get_chat_client()
    context = build_context_blocks(chunks)
    user_msg = f"Context:\n\n{context}\n\nQuestion: {question}"

    try:
        resp = rate_limited_chat_completion(
            client,
            model=settings.generation_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
        )
    except OpenAIError as e:
        raise_clear_error(e)
    answer_text = resp.choices[0].message.content

    cited_indices = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer_text)})
    sources = []
    for idx in cited_indices:
        if 1 <= idx <= len(chunks):
            c = chunks[idx - 1]
            sources.append({"index": idx, "chunk_id": c["chunk_id"], "metadata": c.get("metadata", {})})

    return {
        "question": question,
        "answer": answer_text,
        "cited_indices": cited_indices,
        "sources": sources,
        "context_chunks": chunks,
    }
