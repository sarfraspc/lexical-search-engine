"""Pydantic models for API responses."""

from __future__ import annotations

from pydantic import BaseModel


class SearchResult(BaseModel):
    """One ranked search result entry."""

    document: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    """Search endpoint response payload."""

    results: list[SearchResult]


class IndexResponse(BaseModel):
    """Index rebuild endpoint response payload."""

    message: str
    document_count: int
    vocab_size: int
