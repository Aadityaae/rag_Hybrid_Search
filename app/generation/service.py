"""Phase 3 entry point: RagService.ask(question) runs the full pipeline
end to end -- retrieve, generate, verify citations, score confidence,
and fall back to a structured "don't know" if confidence is too low.
"""
from __future__ import annotations

from openai import OpenAI

from app.generation.citation_verify import verify_citations
from app.generation.confidence import build_dont_know_response, score_answer
from app.generation.prompt import generate_answer
from app.llm_client import get_chat_client
from app.retrieval.engine import HybridRetriever


class RagService:
    def __init__(self, retriever: HybridRetriever = None, client: OpenAI = None):
        self.retriever = retriever or HybridRetriever()
        self._client = client

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = get_chat_client()
        return self._client

    def ask(self, question: str, use_reranker: bool = True, apply_hybrid: bool = True) -> dict:
        retrieval = self.retriever.retrieve(question, use_reranker=use_reranker, apply_hybrid=apply_hybrid)
        final_chunks = retrieval["final_chunks"]

        if not final_chunks:
            fallback = build_dont_know_response(question, final_chunks)
            return {"question": question, "retrieval": retrieval, **fallback,
                    "confidence": {"composite_confidence": 0.0, "below_threshold": True}}

        gen = generate_answer(question, final_chunks, client=self.client)
        citation_report = verify_citations(gen["answer"], final_chunks, client=self.client)
        confidence = score_answer(question, gen["answer"], final_chunks, citation_report, client=self.client)

        result = {
            "question": question,
            "answer": gen["answer"],
            "sources": gen["sources"],
            "citation_report": citation_report,
            "confidence": confidence,
            "retrieval": retrieval,
            "is_fallback": False,
        }

        if confidence["below_threshold"]:
            fallback = build_dont_know_response(question, final_chunks)
            # Keep the original answer visible but signal low trust rather
            # than silently overwriting -- lets the API/UI choose how to show it.
            result["low_confidence_notice"] = fallback

        return result
