"""Configurable chunking. Three strategies, switchable per ingestion run:

  fixed     - fixed-size character windows with overlap (baseline)
  recursive - structure-aware recursive character splitting (section-heading
              boundaries first, then paragraph/sentence fallback)
  semantic  - splits on topic boundaries using embedding-similarity drops
              between consecutive sentence groups

Every Chunk records which strategy produced it so the eval framework can
compare strategies later (Phase 4).
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.ingestion.loaders import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_file: str
    text: str
    chunking_strategy: str
    section_heading: Optional[str] = None
    page_number: Optional[int] = None
    char_count: int = 0
    chunk_index: int = 0

    def __post_init__(self):
        self.char_count = len(self.text)


def chunk_fixed(doc: Document, chunk_size: int = None, overlap: int = None) -> list[Chunk]:
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    text = doc.text
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end]
        if piece.strip():
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()), doc_id=doc.doc_id, source_file=doc.source_file,
                text=piece.strip(), chunking_strategy="fixed",
                section_heading=doc.section_heading, page_number=doc.page_number,
                chunk_index=idx,
            ))
            idx += 1
        start += chunk_size - overlap
    return chunks


def chunk_recursive(doc: Document, chunk_size: int = None, overlap: int = None) -> list[Chunk]:
    """Structure-aware: splits on headings/paragraphs/sentences before
    falling back to raw characters, via LangChain's RecursiveCharacterTextSplitter."""
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(doc.text)
    return [
        Chunk(
            chunk_id=str(uuid.uuid4()), doc_id=doc.doc_id, source_file=doc.source_file,
            text=p.strip(), chunking_strategy="recursive",
            section_heading=doc.section_heading, page_number=doc.page_number,
            chunk_index=i,
        )
        for i, p in enumerate(pieces) if p.strip()
    ]


def _split_sentences(text: str) -> list[str]:
    # Lightweight sentence splitter; good enough for internal docs prose.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_semantic(doc: Document, embed_fn, similarity_drop_threshold: float = 0.25,
                    max_chunk_chars: int = None) -> list[Chunk]:
    """Groups consecutive sentences, embeds each group, and starts a new
    chunk when cosine similarity to the running group drops sharply
    (a topic-boundary signal), or when the max size is hit.

    embed_fn: callable(list[str]) -> np.ndarray of embeddings, injected so
    this module has no direct OpenAI dependency (keeps it unit-testable).
    """
    max_chunk_chars = max_chunk_chars or settings.chunk_size
    sentences = _split_sentences(doc.text)
    if not sentences:
        return []
    if len(sentences) == 1:
        embeddings = None
    else:
        embeddings = embed_fn(sentences)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    chunks: list[Chunk] = []
    current_sentences = [sentences[0]]
    idx = 0

    def flush():
        nonlocal idx
        text = " ".join(current_sentences).strip()
        if text:
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()), doc_id=doc.doc_id, source_file=doc.source_file,
                text=text, chunking_strategy="semantic",
                section_heading=doc.section_heading, page_number=doc.page_number,
                chunk_index=idx,
            ))
            idx += 1

    for i in range(1, len(sentences)):
        candidate_len = len(" ".join(current_sentences)) + len(sentences[i])
        boundary = False
        if embeddings is not None:
            sim = float(np.dot(embeddings[i - 1], embeddings[i]))
            boundary = sim < (1 - similarity_drop_threshold)
        if candidate_len > max_chunk_chars or boundary:
            flush()
            current_sentences = [sentences[i]]
        else:
            current_sentences.append(sentences[i])
    flush()
    return chunks


STRATEGIES = {
    "fixed": chunk_fixed,
    "recursive": chunk_recursive,
    "semantic": chunk_semantic,  # requires embed_fn kwarg
}


def chunk_document(doc: Document, strategy: str = None, embed_fn=None) -> list[Chunk]:
    strategy = strategy or settings.default_chunking_strategy
    if strategy == "semantic":
        if embed_fn is None:
            raise ValueError("semantic chunking requires an embed_fn")
        return chunk_semantic(doc, embed_fn=embed_fn)
    return STRATEGIES[strategy](doc)
