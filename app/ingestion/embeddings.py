"""Local embeddings via sentence-transformers. No API key, no network call
per request, no cost, no rate limits -- the model runs on your own CPU/GPU.

The model weights (~90MB for the default all-MiniLM-L6-v2) download once
from Hugging Face on first use and are cached locally after that.
"""
from __future__ import annotations

import numpy as np

from app.config import settings

_model = None


def get_model():
    """Lazily loads the sentence-transformers model. Kept lazy so importing
    this module doesn't force a model load (and download) before it's
    actually needed."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embeds a list of strings locally. Batches for memory efficiency on
    large corpora; batch size matters far less here than with a paid API
    since there's no per-request cost, just tune for your RAM/CPU."""
    if not texts:
        return np.zeros((0, settings.embedding_dims), dtype=np.float32)
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,  # dedup/vector-store code normalizes where needed
    )
    return embeddings.astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
