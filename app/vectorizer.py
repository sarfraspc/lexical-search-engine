"""Manual TF-IDF vectorization helpers."""

from __future__ import annotations

import math


def build_vocabulary(docs: list[list[str]]) -> dict[str, int]:
    """Build a token-to-index vocabulary from tokenized documents."""
    tokens: set[str] = set()
    for doc in docs:
        tokens.update(doc)
    return {token: idx for idx, token in enumerate(sorted(tokens))}


def term_frequency(doc: list[str], vocab: dict[str, int]) -> list[float]:
    """Compute term frequency vector for a single document."""
    tf: list[float] = [0.0] * len(vocab)
    for token in doc:
        idx = vocab.get(token)
        if idx is not None:
            tf[idx] += 1.0
    return tf


def document_frequency(docs: list[list[str]], vocab: dict[str, int]) -> list[int]:
    """Compute document frequency for each term in the vocabulary."""
    df: list[int] = [0] * len(vocab)
    for doc in docs:
        seen: set[str] = set(doc)
        for token in seen:
            idx = vocab.get(token)
            if idx is not None:
                df[idx] += 1
    return df


def inverse_document_frequency(total_docs: int, df: list[int]) -> list[float]:
    """Compute IDF values using a smoothed formula."""
    idf: list[float] = []
    for freq in df:
        value = math.log((1.0 + total_docs) / (1.0 + freq)) + 1.0
        idf.append(value)
    return idf


def tf_idf_vector(tf: list[float], idf: list[float]) -> list[float]:
    """Create a TF-IDF vector from TF and IDF values."""
    return [tf_val * idf_val for tf_val, idf_val in zip(tf, idf)]


def vectorize_documents(
    docs: list[list[str]],
) -> tuple[dict[str, int], list[float], list[list[float]]]:
    """Vectorize documents into TF-IDF using a shared vocabulary and IDF."""
    vocab = build_vocabulary(docs)
    tfs = [term_frequency(doc, vocab) for doc in docs]
    df = document_frequency(docs, vocab)
    idf = inverse_document_frequency(len(docs), df)
    vectors = [tf_idf_vector(tf, idf) for tf in tfs]
    return vocab, idf, vectors


def vectorize_query(query_tokens: list[str], vocab: dict[str, int], idf: list[float]) -> list[float]:
    """Vectorize a query using the existing vocabulary and IDF."""
    tf = term_frequency(query_tokens, vocab)
    return tf_idf_vector(tf, idf)


def vector_norm(vector: list[float]) -> float:
    """Compute Euclidean norm of a vector."""
    return math.sqrt(sum(value * value for value in vector))
