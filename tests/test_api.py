"""Tests for the FastAPI search API."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from app.api import create_app


def _write_doc(path: Path, name: str, text: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text(text, encoding="utf-8")


def test_search_successful(tmp_path: Path) -> None:
    """Search should return results for valid queries."""
    docs_dir = tmp_path / "documents"
    _write_doc(docs_dir, "a.txt", "Cats are great pets.")
    _write_doc(docs_dir, "b.txt", "Dogs are loyal animals.")

    app = create_app(str(docs_dir))
    with TestClient(app) as client:
        response = client.get("/search", params={"q": "cats"})

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)


def test_search_empty_query(tmp_path: Path) -> None:
    """Empty queries should return 400."""
    docs_dir = tmp_path / "documents"
    _write_doc(docs_dir, "a.txt", "hello world")

    app = create_app(str(docs_dir))
    with TestClient(app) as client:
        response = client.get("/search", params={"q": ""})

    assert response.status_code == 400


def test_search_missing_query_param(tmp_path: Path) -> None:
    """Missing query should return 400."""
    docs_dir = tmp_path / "documents"
    _write_doc(docs_dir, "a.txt", "hello world")

    app = create_app(str(docs_dir))
    with TestClient(app) as client:
        response = client.get("/search")

    assert response.status_code == 400


def test_search_top_k_limit(tmp_path: Path) -> None:
    """Search should return only top 3 results."""
    docs_dir = tmp_path / "documents"
    for i in range(5):
        _write_doc(docs_dir, f"{i}.txt", "cat dog")

    app = create_app(str(docs_dir))
    with TestClient(app) as client:
        response = client.get("/search", params={"q": "cat"})

    data = response.json()
    assert len(data["results"]) == 3


def test_search_response_fields(tmp_path: Path) -> None:
    """Each result should include document, score, and snippet."""
    docs_dir = tmp_path / "documents"
    _write_doc(docs_dir, "a.txt", "cats and dogs")

    app = create_app(str(docs_dir))
    with TestClient(app) as client:
        response = client.get("/search", params={"q": "cats"})

    result = response.json()["results"][0]
    assert "document" in result
    assert "score" in result
    assert "snippet" in result


def test_index_endpoint(tmp_path: Path) -> None:
    """Index endpoint should rebuild and return metadata."""
    docs_dir = tmp_path / "documents"
    _write_doc(docs_dir, "a.txt", "cat dog")

    app = create_app(str(docs_dir))
    with TestClient(app) as client:
        response = client.get("/index")
        _write_doc(docs_dir, "b.txt", "dog fish")
        response2 = client.get("/index")

    data = response.json()
    data2 = response2.json()
    assert data["document_count"] == 1
    assert data["vocab_size"] > 0
    assert data2["document_count"] == 2
