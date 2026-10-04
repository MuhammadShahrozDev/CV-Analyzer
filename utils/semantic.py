from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model():
    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer(MODEL_NAME)
    except Exception:
        return None


def _normalize_texts(texts: Iterable[str]) -> list[str]:
    normalized = [text.strip() for text in texts if text and text.strip()]
    return normalized


# all-MiniLM-L6-v2 truncates at 256 word-pieces, roughly 180-200 words. A full
# resume and a full job description both exceed that, so encoding them whole
# silently compared only their opening paragraphs -- which on a resume is the
# name and contact block. Long text is chunked and mean-pooled instead.
CHUNK_WORDS = 160
CHUNK_OVERLAP = 30


def _chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text]

    chunks: list[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks


def _embed_document(model, text: str) -> np.ndarray:
    """Single unit-length vector for a document of any length."""
    chunks = _chunk_text(text)
    embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.atleast_2d(np.asarray(embeddings, dtype=float))
    pooled = embeddings.mean(axis=0)
    norm = float(np.linalg.norm(pooled))
    if norm > 0:
        pooled = pooled / norm
    return pooled


# Raw cosine is not a percentage. Two genuinely well-matched documents sit
# around 0.5-0.7 with this model, and near-identical text rarely exceeds 0.85,
# so reporting the raw value as a percentage made strong matches look weak.
# These anchors rescale cosine onto a 0-100 reading without lifting the floor:# Re-anchored against measured all-MiniLM-L6-v2 output. Unrelated documents
# sit at ~0.25 cosine rather than 0, because any two English texts share
# grammar and common vocabulary. Treating 0 as the floor handed every resume
# roughly 27 free points. The usable discriminating range is 0.25-0.80.
_EMBEDDING_ANCHORS = ((0.00, 0.0), (0.25, 2.0), (0.40, 18.0), (0.55, 40.0), (0.65, 58.0), (0.75, 78.0), (0.85, 92.0), (1.00, 100.0))
_TFIDF_ANCHORS = ((0.00, 0.0), (0.05, 5.0), (0.15, 30.0), (0.30, 55.0), (0.50, 78.0), (0.70, 92.0), (1.00, 100.0))


def _interpolate(value: float, anchors: tuple[tuple[float, float], ...]) -> float:
    value = max(0.0, min(1.0, value))
    for index in range(len(anchors) - 1):
        low_x, low_y = anchors[index]
        high_x, high_y = anchors[index + 1]
        if low_x <= value <= high_x:
            if high_x == low_x:
                return low_y
            ratio = (value - low_x) / (high_x - low_x)
            return low_y + ratio * (high_y - low_y)
    return anchors[-1][1]


def calibrate_similarity(raw_cosine: float, backend: str = "embedding") -> float:
    anchors = _EMBEDDING_ANCHORS if backend == "embedding" else _TFIDF_ANCHORS
    return round(_interpolate(float(raw_cosine), anchors), 2)


def raw_cosine_similarity(text_a: str, text_b: str) -> tuple[float, str]:
    """Return (raw cosine in 0..1, backend name)."""
    texts = _normalize_texts([text_a, text_b])
    if len(texts) < 2:
        return 0.0, "none"

    model = get_embedding_model()
    if model is not None:
        left = _embed_document(model, texts[0])
        right = _embed_document(model, texts[1])
        return float(np.dot(left, right)), "embedding"

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0]), "tfidf"


def cosine_similarity_score(text_a: str, text_b: str, calibrated: bool = True) -> float:
    """Similarity as a 0-100 reading.

    Chunked so that long documents are compared in full, and calibrated so the
    number means what a reader assumes it means.
    """
    raw, backend = raw_cosine_similarity(text_a, text_b)
    if backend == "none":
        return 0.0
    raw = max(0.0, min(1.0, raw))
    if not calibrated:
        return round(raw * 100, 2)
    return calibrate_similarity(raw, backend)


def sentence_similarity_matrix(source_sentences: list[str], target_sentences: list[str]) -> np.ndarray:
    source_sentences = _normalize_texts(source_sentences)
    target_sentences = _normalize_texts(target_sentences)
    if not source_sentences or not target_sentences:
        return np.zeros((len(source_sentences), len(target_sentences)))

    model = get_embedding_model()
    if model is not None:
        all_sentences = source_sentences + target_sentences
        embeddings = model.encode(all_sentences, normalize_embeddings=True, show_progress_bar=False)
        source_embeddings = embeddings[: len(source_sentences)]
        target_embeddings = embeddings[len(source_sentences) :]
        return np.asarray(np.matmul(source_embeddings, target_embeddings.T))

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(source_sentences + target_sentences)
    source_matrix = matrix[: len(source_sentences)]
    target_matrix = matrix[len(source_sentences) :]
    return cosine_similarity(source_matrix, target_matrix)


def all_best_matches(source_sentences: list[str], target_sentences: list[str]) -> list[dict[str, object]]:
    """Best target for every source sentence, regardless of threshold.

    The original helper returned only sentences that cleared the threshold, so
    callers could not tell "five duties, none matched" from "no duties found".
    Both produced an empty list and therefore a score of zero. Returning every
    source lets the caller compute genuine coverage.
    """
    sources = _normalize_texts(source_sentences)
    targets = _normalize_texts(target_sentences)
    if not sources or not targets:
        return []

    matrix = sentence_similarity_matrix(sources, targets)
    if matrix.size == 0:
        return []

    results: list[dict[str, object]] = []
    for source_index, source_sentence in enumerate(sources):
        best_target_index = int(np.argmax(matrix[source_index]))
        raw = float(matrix[source_index, best_target_index])
        results.append(
            {
                "source": source_sentence,
                "target": targets[best_target_index],
                "raw": round(raw, 4),
                "score": round(max(0.0, min(1.0, raw)) * 100, 2),
            }
        )
    return results


def best_sentence_matches(source_sentences: list[str], target_sentences: list[str], threshold: float = 0.55) -> list[dict[str, object]]:
    """Backward-compatible view: only matches at or above the threshold."""
    return [match for match in all_best_matches(source_sentences, target_sentences) if float(match["raw"]) >= threshold]
