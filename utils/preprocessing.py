from __future__ import annotations

import re
from functools import lru_cache

import spacy


@lru_cache(maxsize=1)
def get_nlp():
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = spacy.blank("en")
    if nlp.max_length < 2_000_000:
        nlp.max_length = 2_000_000
    return nlp


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text: str) -> str:
    if not text:
        return ""

    nlp = get_nlp()
    lowered = text.lower()
    doc = nlp(lowered)
    tokens: list[str] = []

    for token in doc:
        if token.is_space or token.is_punct or token.is_stop:
            continue
        if not token.text.strip():
            continue
        if token.like_num:
            continue
        lemma = token.lemma_.strip().lower() if token.lemma_ and token.lemma_ != "-pron-" else token.text.strip().lower()
        lemma = re.sub(r"[^a-z0-9+#.-]", "", lemma)
        if lemma:
            tokens.append(lemma)

    return normalize_whitespace(" ".join(tokens))
