"""API schemas."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    data_root: Optional[str] = Field(default="data", description="Path to data root")


class EmbedRequest(BaseModel):
    texts: List[str]


class QueryRequest(BaseModel):
    query_text: str = Field(..., description="Natural language query")
    image_b64: Optional[str] = Field(default=None, description="Optional base64 image")
    top_k: int = Field(default=5, ge=1, le=10)
    generator_backend: Optional[str] = Field(default=None)


class ContextModel(BaseModel):
    source_id: str
    content: str
    modality: str
    metadata: dict
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    contexts: List[ContextModel]
    modality_breakdown: dict
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
