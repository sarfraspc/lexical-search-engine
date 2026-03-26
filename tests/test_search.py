"""Tests for cosine similarity and ranking."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
   
import math

from app.search import cosine_similarity, rank_documents, top_k_results


def test_cosine_similarity_identical() -> None:
    """Identical vectors should have similarity 1."""
    vec = [1.0, 2.0, 3.0]
    score = cosine_similarity(vec, vec)
    assert math.isclose(score, 1.0)


def test_cosine_similarity_orthogonal() -> None:
    """Orthogonal vectors should have similarity 0."""
    score = cosine_similarity([1.0, 0.0], [0.0, 2.0])
    assert math.isclose(score, 0.0)


def test_cosine_similarity_zero_vector() -> None:
    """Zero vectors should return 0 similarity."""
    score = cosine_similarity([0.0, 0.0], [1.0, 2.0])
    assert math.isclose(score, 0.0)


def test_ranking_correctness() -> None:
    """Documents should be ranked by descending similarity."""
    query = [1.0, 0.0]
    docs = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
    ranked = rank_documents(query, docs)
    assert [idx for idx, _ in ranked] == [0, 1, 2]


def test_top_k_behavior() -> None:
    """Top-k should return only k items in order."""
    query = [1.0, 0.0]
    docs = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
    top = top_k_results(query, docs, k=2)
    assert len(top) == 2
    assert [idx for idx, _ in top] == [0, 1]
