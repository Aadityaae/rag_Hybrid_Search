"""BM25 sparse keyword index. Persisted alongside the vector store and
built over the exact same chunk set / chunk_id namespace so dense and
sparse results can be fused by id in the RRF layer."""
from __future__ import annotations

import pickle
import re

from rank_bm25 import BM25Okapi

from app.config import BM25_INDEX_PATH


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_./-]+", text.lower())


class BM25Store:
    def __init__(self):
        self.chunk_ids: list[str] = []
        self.texts: list[str] = []
        self.metadatas: list[dict] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunk_ids: list[str], texts: list[str], metadatas: list[dict]):
        self.chunk_ids = list(chunk_ids)
        self.texts = list(texts)
        self.metadatas = list(metadatas)
        tokenized = [_tokenize(t) for t in self.texts]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def add(self, chunk_ids: list[str], texts: list[str], metadatas: list[dict]):
        """Rebuilds the index including new chunks. BM25Okapi has no
        incremental-add API, so this recomputes over the full corpus --
        cheap enough at the chunk counts this project targets."""
        self.chunk_ids.extend(chunk_ids)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        tokenized = [_tokenize(t) for t in self.texts]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def query(self, query_text: str, top_k: int) -> list[dict]:
        if self._bm25 is None or not self.texts:
            return []
        scores = self._bm25.get_scores(_tokenize(query_text))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                "chunk_id": self.chunk_ids[i],
                "text": self.texts[i],
                "metadata": self.metadatas[i],
                "score": float(scores[i]),
            }
            for i in ranked_idx if scores[i] > 0
        ]

    def save(self, path=None):
        path = path or BM25_INDEX_PATH
        with open(path, "wb") as f:
            pickle.dump({"chunk_ids": self.chunk_ids, "texts": self.texts, "metadatas": self.metadatas}, f)

    def load(self, path=None) -> bool:
        path = path or BM25_INDEX_PATH
        if not path.exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.build(data["chunk_ids"], data["texts"], data["metadatas"])
        return True

    def count(self) -> int:
        return len(self.chunk_ids)
