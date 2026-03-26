"""Tests for document indexing."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.indexer import build_index
from app.vectorizer import vector_norm


def _write_doc(path: Path, name: str, text: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text(text, encoding="utf-8")


def test_build_index_basic(tmp_path: Path) -> None:
    """Index should include filename, text, tokens, vector, norm, snippet."""
    docs_dir = tmp_path / "documents"
    _write_doc(docs_dir, "a.txt", "Hello world")
    _write_doc(docs_dir, "b.txt", "World of cats")

    index = build_index(str(docs_dir))
    documents = index["documents"]

    assert len(documents) == 2
    for doc in documents:
        assert "filename" in doc
        assert "text" in doc
        assert "tokens" in doc
        assert "vector" in doc
        assert "norm" in doc
        assert "snippet" in doc


def test_index_vectors_and_norms(tmp_path: Path) -> None:
    """Vector length and norm should be consistent with vocabulary."""
    docs_dir = tmp_path / "documents"
    _write_doc(docs_dir, "a.txt", "cat dog")
    _write_doc(docs_dir, "b.txt", "dog fish")

    index = build_index(str(docs_dir))
    vocab = index["vocab"]
    documents = index["documents"]

    assert len(vocab) > 0
    for doc in documents:
        vector = doc["vector"]
        norm = doc["norm"]
        assert len(vector) == len(vocab)
        assert norm == vector_norm(vector)


def test_build_index_ignores_non_txt_files(tmp_path: Path) -> None:
    """Indexer should load only .txt files."""
    docs_dir = tmp_path / "documents"
    _write_doc(docs_dir, "a.txt", "hello world")
    (docs_dir / "ignore.md").write_text("should not be loaded", encoding="utf-8")

    index = build_index(str(docs_dir))
    documents = index["documents"]

    assert len(documents) == 1
    assert documents[0]["filename"] == "a.txt"


def test_snippet_is_shortened(tmp_path: Path) -> None:
    """Snippet should be capped at 200 characters."""
    docs_dir = tmp_path / "documents"
    long_text = "a" * 250
    _write_doc(docs_dir, "long.txt", long_text)

    index = build_index(str(docs_dir))
    snippet = index["documents"][0]["snippet"]

    assert len(snippet) == 200
