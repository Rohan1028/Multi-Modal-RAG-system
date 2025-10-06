"""FastAPI application entrypoint."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.app import deps, schemas
from src.app.settings import get_settings
from src.generation.cite_guard import fill_missing_sources
from src.generation.generator import get_generator
from src.ingestion.indexer import IndexSummary, build_indexes
from src.retrieval.hybrid import Context, RetrievalResult
from src.utils.timer import track_time

app = FastAPI(title="Multimodal RAG System", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=schemas.HealthResponse)
def health() -> schemas.HealthResponse:
    return schemas.HealthResponse(status="ok")


@app.post("/index")
def index_data(request: schemas.IndexRequest) -> IndexSummary:
    settings = get_settings()
    data_root = Path(request.data_root or "data")
    summary = build_indexes(data_root=data_root, index_dir=settings.index_dir, duckdb_path=settings.duckdb_path)
    deps.get_retriever.cache_clear()  # type: ignore[attr-defined]
    return summary


@app.post("/embed")
def embed(request: schemas.EmbedRequest) -> dict:
    retriever = deps.get_retriever()
    vectors = retriever.text_embedder.encode(request.texts).tolist()
    return {"embeddings": vectors}


@app.post("/query", response_model=schemas.QueryResponse)
def query(request: schemas.QueryRequest) -> schemas.QueryResponse:
    settings = get_settings()
    retriever = deps.get_retriever()
    generator = get_generator(request.generator_backend or settings.generator_backend)

    timings: dict[str, float] = {}
    start = time.perf_counter()
    with track_time(timings, "retrieval_ms"):
        retrieval: RetrievalResult = retriever.retrieve(
            query_text=request.query_text,
            top_k=request.top_k,
            image_b64=request.image_b64,
        )
    contexts: List[Context] = retrieval["contexts"]
    modality_breakdown: Dict[str, int] = retrieval["modality_breakdown"]

    if not contexts:
        raise HTTPException(status_code=404, detail="No context found for query")

    with track_time(timings, "generation_ms"):
        result = generator.generate(request.query_text, contexts)

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    citations = result.citations or [ctx.source_id for ctx in contexts]
    answer = fill_missing_sources(result.answer, citations)

    context_models = [
        schemas.ContextModel(
            source_id=ctx.source_id,
            content=ctx.content,
            modality=ctx.modality,
            metadata=ctx.metadata,
            score=ctx.score,
        )
        for ctx in contexts
    ]

    return schemas.QueryResponse(
        answer=answer,
        citations=citations,
        contexts=context_models,
        modality_breakdown=modality_breakdown,
        latency_ms=latency_ms,
    )
