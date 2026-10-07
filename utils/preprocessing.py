from __future__ import annotations

import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = re.sub(
        r"[ \t\f\v]+",
        " ",
        text,
    )

    text = re.sub(
        r" *\n *",
        "\n",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def normalize_unicode(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
        "\u2022": " ",
        "\u25cf": " ",
        "\u25aa": " ",
        "\u25e6": " ",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new,
        )

    return text


def preprocess_text(text: str) -> str:
    if not text:
        return ""

    text = normalize_unicode(
        text
    )

    text = text.lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
    )

    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"[^a-z0-9+#./\-\s]",
        " ",
        text,
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n+",
        " ",
        text,
    )

    return text.strip()
