"""FastAPI application for lexical search."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app.indexer import build_index
from app.models import IndexResponse, SearchResponse
from app.preprocess import preprocess_text
from app.search import top_k_results
from app.vectorizer import vectorize_query


def create_app(documents_dir: str = "documents") -> FastAPI:
    """Create a FastAPI app and build the index once at startup."""
    app = FastAPI()
    static_dir = Path(__file__).resolve().parent / "static"

    @app.on_event("startup")
    def _startup() -> None:
        app.state.index = build_index(documents_dir)

    @app.get("/search", response_model=SearchResponse)
    def search(q: str | None = None) -> SearchResponse:
        """Search indexed documents by query string."""
        if q is None or not q.strip():
            raise HTTPException(status_code=400, detail="Query is required.")

        if not hasattr(app.state, "index"):
            app.state.index = build_index(documents_dir)
        index = app.state.index
        tokens = preprocess_text(q)
        if not tokens:
            raise HTTPException(status_code=400, detail="Query must contain valid terms.")

        vocab = index["vocab"]
        idf = index["idf"]
        documents = index["documents"]
        doc_vectors = [doc["vector"] for doc in documents]

        query_vec = vectorize_query(tokens, vocab, idf)
        ranked = top_k_results(query_vec, doc_vectors, k=3)

        results: list[dict[str, object]] = []
        for doc_index, score in ranked:
            doc = documents[doc_index]
            results.append(
                {
                    "document": doc["filename"],
                    "score": score,
                    "snippet": doc["snippet"],
                }
            )

        return SearchResponse(results=results)

    @app.get("/index", response_model=IndexResponse)
    def index_info() -> IndexResponse:
        """Rebuild the index and return basic metadata."""
        index = build_index(documents_dir)
        app.state.index = index
        return IndexResponse(
            message="Index rebuilt successfully.",
            document_count=len(index["documents"]),
            vocab_size=len(index["vocab"]),
        )

    @app.get("/")
    def home() -> FileResponse:
        """Serve the minimal HTML search UI."""
        return FileResponse(static_dir / "index.html")

    return app


app = create_app()
