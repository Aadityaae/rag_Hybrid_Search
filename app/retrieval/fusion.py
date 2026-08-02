"""Reciprocal Rank Fusion (RRF). Combines a dense-ranked list and a
sparse-ranked list into one ranked list using rank position (not raw
score, since cosine similarity and BM25 scores live on different scales
and aren't directly comparable). Weighting is configurable so the
dense/sparse balance can be tuned per use case.
"""
from __future__ import annotations

from app.config import settings


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    dense_weight: float = None,
    sparse_weight: float = None,
    rrf_k: int = None,
) -> list[dict]:
    dense_weight = settings.fusion_dense_weight if dense_weight is None else dense_weight
    sparse_weight = settings.fusion_sparse_weight if sparse_weight is None else sparse_weight
    rrf_k = rrf_k or settings.rrf_k

    scores: dict[str, float] = {}
    payload: dict[str, dict] = {}
    sources: dict[str, set] = {}

    for rank, r in enumerate(dense_results):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + dense_weight * (1.0 / (rrf_k + rank + 1))
        payload.setdefault(cid, r)
        sources.setdefault(cid, set()).add("dense")

    for rank, r in enumerate(sparse_results):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + sparse_weight * (1.0 / (rrf_k + rank + 1))
        payload.setdefault(cid, r)
        sources.setdefault(cid, set()).add("sparse")

    fused = []
    for cid, score in scores.items():
        item = dict(payload[cid])
        item["fusion_score"] = score
        item["matched_by"] = sorted(sources[cid])
        fused.append(item)

    fused.sort(key=lambda x: x["fusion_score"], reverse=True)
    return fused
