"""Search helpers for ranking documents with cosine similarity."""

from __future__ import annotations
from app.vectorizer import vector_norm


def cosine_similarity(
    vec1: list[float],
    vec2: list[float],
    norm1: float | None = None,
    norm2: float | None = None,
) -> float:
    """Compute cosine similarity between two vectors with zero guards."""
    if norm1 is None:
        norm1 = vector_norm(vec1)
    if norm2 is None:
        norm2 = vector_norm(vec2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    return dot / (norm1 * norm2)


def rank_documents(query_vec: list[float], doc_vectors: list[list[float]]) -> list[tuple[int, float]]:
    """Rank documents by similarity to the query vector."""
    query_norm = vector_norm(query_vec)
    ranked: list[tuple[int, float]] = []
    for idx, doc_vec in enumerate(doc_vectors):
        score = cosine_similarity(query_vec, doc_vec, norm1=query_norm)
        ranked.append((idx, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def top_k_results(
    query_vec: list[float],
    doc_vectors: list[list[float]],
    k: int,
) -> list[tuple[int, float]]:
    """Return top-k ranked documents."""
    if k <= 0:
        return []
    ranked = rank_documents(query_vec, doc_vectors)
    return ranked[:k]
