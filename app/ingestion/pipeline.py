"""Orchestrates Phase 1: load raw docs, chunk them, embed, dedup, and
write to BOTH the vector store and BM25 index so they never drift out
of sync (same chunk_id list is used for both inserts)."""
from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.ingestion.chunking import chunk_document, Chunk
from app.ingestion.dedup import DuplicateFilter
from app.ingestion.embeddings import embed_texts
from app.ingestion.loaders import load_directory, persist_processed, Document
from app.retrieval.bm25_store import BM25Store
from app.retrieval.vector_store import VectorStore


def _chunk_to_metadata(chunk: Chunk) -> dict:
    return {
        "source_file": chunk.source_file,
        "doc_id": chunk.doc_id,
        "chunk_index": chunk.chunk_index,
        "section_heading": chunk.section_heading or "",
        "page_number": chunk.page_number or 0,
        "chunking_strategy": chunk.chunking_strategy,
        "char_count": chunk.char_count,
    }


def ingest_directory(
    raw_dir: Path,
    strategy: str = None,
    vector_store: VectorStore = None,
    bm25_store: BM25Store = None,
) -> dict:
    strategy = strategy or settings.default_chunking_strategy
    vector_store = vector_store or VectorStore()
    bm25_store = bm25_store or BM25Store()
    bm25_store.load()  # merge with any existing index rather than clobber

    docs: list[Document] = load_directory(raw_dir)
    persist_processed(docs)

    all_chunks: list[Chunk] = []
    for doc in docs:
        embed_fn = (lambda texts: embed_texts(texts)) if strategy == "semantic" else None
        all_chunks.extend(chunk_document(doc, strategy=strategy, embed_fn=embed_fn))

    if not all_chunks:
        return {"documents": len(docs), "chunks_embedded": 0, "chunks_deduped": 0}

    embeddings = embed_texts([c.text for c in all_chunks])

    dedup = DuplicateFilter()
    kept_chunks, kept_embeddings, dropped_ids = dedup.filter_chunks(all_chunks, embeddings)

    ids = [c.chunk_id for c in kept_chunks]
    texts = [c.text for c in kept_chunks]
    metadatas = [_chunk_to_metadata(c) for c in kept_chunks]

    vector_store.add(ids=ids, embeddings=kept_embeddings, documents=texts, metadatas=metadatas)
    bm25_store.add(chunk_ids=ids, texts=texts, metadatas=metadatas)
    bm25_store.save()

    return {
        "documents": len(docs),
        "chunks_total": len(all_chunks),
        "chunks_embedded": len(kept_chunks),
        "chunks_deduped": len(dropped_ids),
        "strategy": strategy,
        "vector_store_count": vector_store.count(),
        "bm25_count": bm25_store.count(),
    }
