from pydantic import BaseModel, Field
from typing import List, Optional


class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language query for code search")
    top_k: int = Field(10, description="Number of results to return", ge=1, le=100)
    language: Optional[str] = Field(None, description="Filter by programming language")


class CodeResult(BaseModel):
    rank: int
    code: str
    language: str
    func_name: str
    docstring: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: List[CodeResult]
    total_results: int


class EmbedRequest(BaseModel):
    code: str = Field(..., description="Code snippet to embed")
    language: str = Field("python", description="Programming language")


class EmbedResponse(BaseModel):
    embedding: List[float]
    dimension: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    index_size: int