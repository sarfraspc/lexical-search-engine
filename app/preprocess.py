"""Text preprocessing helpers for lexical search."""

from __future__ import annotations

import re

STOPWORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def _normalize_text(text: str) -> str:
    """Lowercase and remove punctuation-like characters."""
    lowered: str = text.lower()
    cleaned: str = re.sub(r"[^\w\s]", " ", lowered)
    return cleaned


def _tokenize(text: str) -> list[str]:
    """Split text into word tokens."""
    return re.findall(r"\b\w+\b", text)


def preprocess_text(text: str) -> list[str]:
    """Preprocess input text and return filtered tokens."""
    normalized: str = _normalize_text(text)
    tokens: list[str] = _tokenize(normalized)
    return [token for token in tokens if token not in STOPWORDS]
