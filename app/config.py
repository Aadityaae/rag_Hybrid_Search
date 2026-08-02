"""Central configuration. All tunables live here so retrieval/chunking
experiments don't require touching business logic."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
CHROMA_DIR = ROOT_DIR / "data" / "chroma"
BM25_INDEX_PATH = ROOT_DIR / "data" / "bm25_index.pkl"
GOLDEN_DATASET_PATH = ROOT_DIR / "app" / "eval" / "golden_qa.json"

for d in (RAW_DIR, PROCESSED_DIR, CHROMA_DIR):
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    # Generation + LLM-judge provider: Groq's free tier, OpenAI-compatible API.
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    generation_model: str = os.getenv("GENERATION_MODEL", "llama-3.3-70b-versatile")
    llm_judge_model: str = os.getenv("LLM_JUDGE_MODEL", "llama-3.1-8b-instant")
    # Kept conservatively under Groq's free-tier 30 RPM/model cap so the
    # pipeline throttles itself proactively instead of bursting into 429s.
    groq_requests_per_minute: int = int(os.getenv("GROQ_RPM_LIMIT", "25"))

    # Embeddings: local sentence-transformers model, no API key or cost.
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_dims: int = 384  # must match the embedding_model above

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120
    default_chunking_strategy: str = "recursive"  # fixed | recursive | semantic

    # Dedup
    dedup_similarity_threshold: float = 0.95

    # Retrieval
    dense_top_k: int = 10
    sparse_top_k: int = 10
    fusion_dense_weight: float = 0.7
    fusion_sparse_weight: float = 0.3
    rrf_k: int = 60  # RRF damping constant
    rerank_candidate_pool: int = 20
    rerank_top_n: int = 5

    # Generation / confidence
    min_retrieval_confidence: float = 0.35
    collection_name: str = "internal_docs"


settings = Settings()
