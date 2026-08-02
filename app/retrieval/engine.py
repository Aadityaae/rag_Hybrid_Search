"""Phase 2 entry point: HybridRetriever.retrieve(question) runs dense +
sparse retrieval, fuses with RRF, reranks the top candidates, and returns
the final top-k chunks used for generation. Also exposes dense-only
retrieval so the dashboard can show a side-by-side comparison.
"""
from __future__ import annotations

from app.config import settings
from app.ingestion.embeddings import embed_query
from app.retrieval.bm25_store import BM25Store
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import rerank
from app.retrieval.vector_store import VectorStore


class HybridRetriever:
    def __init__(self, vector_store: VectorStore = None, bm25_store: BM25Store = None):
        self.vector_store = vector_store or VectorStore()
        self.bm25_store = bm25_store or BM25Store()
        if self.bm25_store.count() == 0:
            self.bm25_store.load()

    def dense_only(self, question: str, top_k: int = None) -> list[dict]:
        top_k = top_k or settings.dense_top_k
        q_emb = embed_query(question)
        return self.vector_store.query(q_emb, top_k=top_k)

    def sparse_only(self, question: str, top_k: int = None) -> list[dict]:
        top_k = top_k or settings.sparse_top_k
        return self.bm25_store.query(question, top_k=top_k)

    def retrieve(self, question: str, use_reranker: bool = True, apply_hybrid: bool = True) -> dict:
        dense_results = self.dense_only(question)
        sparse_results = self.sparse_only(question) if apply_hybrid else []

        fused = reciprocal_rank_fusion(dense_results, sparse_results) if apply_hybrid else dense_results
        candidate_pool = fused[: settings.rerank_candidate_pool]

        final = rerank(question, candidate_pool) if use_reranker else candidate_pool[: settings.rerank_top_n]

        return {
            "question": question,
            "dense_results": dense_results,
            "sparse_results": sparse_results,
            "fused_results": fused,
            "final_chunks": final,
        }
