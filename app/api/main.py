"""Phase 5: FastAPI service.

POST /v1/ask       - question -> answer + citations + confidence
GET  /v1/documents - list indexed documents
POST /v1/ingest     - ingest a directory of new documents
GET  /health        - liveness check
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import RAW_DIR
from app.generation.service import RagService
from app.ingestion.loaders import load_processed
from app.ingestion.pipeline import ingest_directory
from app.retrieval.engine import HybridRetriever

app = FastAPI(
    title="Internal Docs RAG API",
    description="Hybrid-search RAG over internal documentation with grounded citations.",
    version="1.0.0",
)

_retriever = HybridRetriever()
_service = RagService(retriever=_retriever)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    use_reranker: bool = True
    apply_hybrid: bool = True


class IngestRequest(BaseModel):
    directory: str = Field(default=str(RAW_DIR), description="Path to a directory of docs to ingest")
    strategy: str = Field(default="recursive", pattern="^(fixed|recursive|semantic)$")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/ask")
def ask(req: AskRequest):
    try:
        result = _service.ask(req.question, use_reranker=req.use_reranker, apply_hybrid=req.apply_hybrid)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return result


@app.get("/v1/documents")
def list_documents():
    try:
        docs = load_processed()
    except FileNotFoundError:
        return {"documents": []}
    seen = {}
    for d in docs:
        seen.setdefault(d.source_file, 0)
        seen[d.source_file] += 1
    return {"documents": [{"source_file": k, "sections": v} for k, v in seen.items()]}


@app.post("/v1/ingest")
def ingest(req: IngestRequest):
    directory = Path(req.directory)
    if not directory.exists():
        raise HTTPException(status_code=400, detail=f"Directory not found: {directory}")
    try:
        stats = ingest_directory(directory, strategy=req.strategy,
                                  vector_store=_retriever.vector_store, bm25_store=_retriever.bm25_store)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return stats
