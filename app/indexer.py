"""Document indexing helpers."""

from __future__ import annotations

from pathlib import Path

from app.preprocess import preprocess_text
from app.vectorizer import vector_norm, vectorize_documents


def _load_text_files(documents_dir: str) -> list[tuple[str, str]]:
    """Load all .txt files from a directory."""
    path = Path(documents_dir)
    files = sorted(path.glob("*.txt"))
    items: list[tuple[str, str]] = []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        items.append((file_path.name, text))
    return items


def _make_snippet(text: str, max_length: int = 200) -> str:
    """Create a short preview snippet: first 2 lines or 200 chars."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 2:
        snippet = " ".join(lines[:2])
    elif lines:
        snippet = lines[0]
    else:
        snippet = " ".join(text.split())
    return snippet[:max_length]


def build_index(documents_dir: str = "documents") -> dict[str, object]:
    """Build an in-memory index from text files."""
    items = _load_text_files(documents_dir)
    filenames = [name for name, _ in items]
    texts = [text for _, text in items]
    tokens_list = [preprocess_text(text) for text in texts]

    vocab, idf, vectors = vectorize_documents(tokens_list)
    norms = [vector_norm(vec) for vec in vectors]

    documents: list[dict[str, object]] = []
    for name, text, tokens, vector, norm in zip(
        filenames, texts, tokens_list, vectors, norms
    ):
        documents.append(
            {
                "filename": name,
                "text": text,
                "tokens": tokens,
                "vector": vector,
                "norm": norm,
                "snippet": _make_snippet(text),
            }
        )

    return {
        "vocab": vocab,
        "idf": idf,
        "documents": documents,
    }
