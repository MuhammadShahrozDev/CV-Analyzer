from __future__ import annotations

import re
from functools import lru_cache


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being",
    "but", "by", "for", "from", "had", "has", "have", "he", "her",
    "hers", "him", "his", "i", "if", "in", "into", "is", "it",
    "its", "me", "my", "of", "on", "or", "our", "ours", "she",
    "so", "that", "the", "their", "theirs", "them", "they", "this",
    "to", "was", "we", "were", "what", "when", "where", "which",
    "who", "will", "with", "you", "your", "yours"
}


@lru_cache(maxsize=1)
def get_nlp():
    return None


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_token(token: str) -> str:
    token = token.strip().lower()
    token = re.sub(r"[^a-z0-9+#.-]", "", token)

    if not token:
        return ""

    if token.isdigit():
        return ""

    if token in STOP_WORDS:
        return ""

    return token


def preprocess_text(text: str) -> str:
    if not text:
        return ""

    lowered = text.lower()

    raw_tokens = re.findall(
        r"[a-z0-9+#.-]+",
        lowered
    )

    tokens: list[str] = []

    for token in raw_tokens:
        cleaned = normalize_token(token)

        if cleaned:
            tokens.append(cleaned)

    return normalize_whitespace(" ".join(tokens))