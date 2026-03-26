"""Tests for text preprocessing."""

from app.preprocess import preprocess_text


def test_preprocess_lowercases_words() -> None:
    """It lowercases all tokens."""
    result = preprocess_text("HELLO World")
    assert result == ["hello", "world"]


def test_preprocess_removes_punctuation() -> None:
    """It removes punctuation and keeps words."""
    result = preprocess_text("Hello, world! This is great.")
    assert result == ["hello", "world", "this", "great"]


def test_preprocess_removes_stopwords() -> None:
    """It removes configured stopwords."""
    result = preprocess_text("The cat and the dog are in the house")
    assert result == ["cat", "dog", "house"]


def test_preprocess_handles_empty_string() -> None:
    """It returns an empty list for empty input."""
    result = preprocess_text("")
    assert result == []
