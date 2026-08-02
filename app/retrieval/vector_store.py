"""ChromaDB-backed dense vector store. One collection holds all chunks;
metadata carries everything needed to reconstruct citations and to keep
the BM25 index in sync (same chunk_id namespace)."""
from __future__ import annotations

import chromadb

from app.config import settings, CHROMA_DIR


class VectorStore:
    def __init__(self, collection_name: str = None):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name=collection_name or settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids: list[str], embeddings, documents: list[str], metadatas: list[dict]):
        if not ids:
            return
        self.collection.add(ids=ids, embeddings=embeddings.tolist(), documents=documents, metadatas=metadatas)

    def query(self, query_embedding, top_k: int) -> list[dict]:
        res = self.collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)
        out = []
        for i in range(len(res["ids"][0])):
            out.append({
                "chunk_id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i],
                "score": 1 - res["distances"][0][i],  # cosine similarity
            })
        return out

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name, metadata={"hnsw:space": "cosine"},
        )
