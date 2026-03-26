"""Tests for manual TF-IDF vectorization."""

import math

from app.vectorizer import (
    build_vocabulary,
    document_frequency,
    inverse_document_frequency,
    term_frequency,
    tf_idf_vector,
    vector_norm,
    vectorize_documents,
    vectorize_query,
)


def test_build_vocabulary_sorted() -> None:
    """Vocabulary should be deterministic and sorted."""
    docs = [["dog", "cat"], ["cat", "fish"]]
    vocab = build_vocabulary(docs)
    assert list(vocab.keys()) == ["cat", "dog", "fish"]


def test_term_frequency_counts() -> None:
    """TF should count occurrences per document."""
    vocab = {"cat": 0, "dog": 1}
    tf = term_frequency(["dog", "cat", "dog"], vocab)
    assert tf == [1.0, 2.0]


def test_document_frequency_counts() -> None:
    """DF should count documents containing each term."""
    docs = [["cat", "dog"], ["cat"], ["dog", "fish"]]
    vocab = build_vocabulary(docs)
    df = document_frequency(docs, vocab)
    assert df == [2, 2, 1]  # cat, dog, fish


def test_idf_is_smoothed() -> None:
    """IDF should be smoothed and higher for rarer terms."""
    df = [2, 1]
    idf = inverse_document_frequency(3, df)
    assert idf[1] > idf[0]


def test_tf_idf_vector_multiplies() -> None:
    """TF-IDF should multiply TF by IDF elementwise."""
    tf = [1.0, 2.0]
    idf = [2.0, 3.0]
    tfidf = tf_idf_vector(tf, idf)
    assert tfidf == [2.0, 6.0]


def test_vectorize_documents_shapes() -> None:
    """Vectorization should return one vector per document."""
    docs = [["cat", "dog"], ["dog", "fish"]]
    vocab, idf, vectors = vectorize_documents(docs)
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vocab)
    assert len(idf) == len(vocab)


def test_vectorize_query_uses_vocab() -> None:
    """Query vector should align with vocabulary and IDF."""
    docs = [["cat", "dog"], ["dog", "fish"]]
    vocab, idf, doc_vectors = vectorize_documents(docs)
    query_vec = vectorize_query(["dog", "unknown"], vocab, idf)
    assert len(query_vec) == len(vocab)
    dog_index = vocab["dog"]
    assert query_vec[dog_index] > 0.0


def test_vector_norm() -> None:
    """Vector norm should compute Euclidean length."""
    norm = vector_norm([3.0, 4.0])
    assert math.isclose(norm, 5.0)


def test_vectorize_documents_empty() -> None:
    """Empty input should return empty structures."""
    vocab, idf, vectors = vectorize_documents([])
    assert vocab == {}
    assert idf == []
    assert vectors == []


def test_query_unknown_words() -> None:
    """Query with only unknown words should be all zeros."""
    docs = [["cat", "dog"]]
    vocab, idf, vectors = vectorize_documents(docs)
    query_vec = vectorize_query(["unknown", "missing"], vocab, idf)
    assert query_vec == [0.0] * len(vocab)
