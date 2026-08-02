"""Near-duplicate detection. Before a chunk is inserted into the index,
check cosine similarity against already-inserted chunk embeddings; skip
(and flag) anything above the similarity threshold so the retriever never
wastes top-k slots on redundant content repeated across source docs.
"""
from __future__ import annotations

import numpy as np

from app.config import settings


class DuplicateFilter:
    """Stateful filter: call `is_duplicate` in insertion order. Maintains
    a running matrix of accepted embeddings for comparison. O(n) per check
    which is fine for the corpus sizes this project targets (thousands of
    chunks); swap for an ANN index if that stops being true."""

    def __init__(self, threshold: float = None):
        self.threshold = threshold or settings.dedup_similarity_threshold
        self._accepted: list[np.ndarray] = []
        self.duplicate_log: list[dict] = []

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec if norm == 0 else vec / norm

    def is_duplicate(self, chunk_id: str, embedding: np.ndarray) -> tuple[bool, float]:
        if not self._accepted:
            self._accepted.append(self._normalize(embedding))
            return False, 0.0
        emb_n = self._normalize(embedding)
        sims = np.array([float(np.dot(emb_n, a)) for a in self._accepted])
        max_sim = float(sims.max())
        if max_sim > self.threshold:
            self.duplicate_log.append({"chunk_id": chunk_id, "max_similarity": max_sim})
            return True, max_sim
        self._accepted.append(emb_n)
        return False, max_sim

    def filter_chunks(self, chunks: list, embeddings: np.ndarray) -> tuple[list, np.ndarray, list]:
        """Returns (kept_chunks, kept_embeddings, dropped_chunk_ids)."""
        kept_chunks, kept_vecs, dropped = [], [], []
        for chunk, emb in zip(chunks, embeddings):
            dup, sim = self.is_duplicate(chunk.chunk_id, emb)
            if dup:
                dropped.append(chunk.chunk_id)
            else:
                kept_chunks.append(chunk)
                kept_vecs.append(emb)
        kept_arr = np.array(kept_vecs, dtype=np.float32) if kept_vecs else np.zeros((0, embeddings.shape[1]))
        return kept_chunks, kept_arr, dropped
