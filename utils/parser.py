from __future__ import annotations

import asyncio
import importlib.util
import re
import tempfile
import uuid
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pypdf import PdfReader
from docx import Document
from fastapi import HTTPException, UploadFile

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional fallback
    fitz = None

try:
    import trafilatura  # Clean "main content" extraction, same job static pages use.
except Exception:  # pragma: no cover - optional dependency
    trafilatura = None

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional dependency
    BeautifulSoup = None

ALLOWED_RESUME_EXTENSIONS = {".pdf", ".docx"}
ALLOWED_JOB_EXTENSIONS = {".pdf", ".docx", ".txt"}

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Below this many words, a fetched page is treated as a JS-rendered shell (nav,
# cookie banner, "please enable JavaScript" placeholder) rather than the real
# posting, and the next extraction strategy is tried.
MIN_USABLE_WORDS = 60

# A rendered page can come back technically non-empty but still be nothing but
# navigation, header chips and a sign-in prompt - which is what a JavaScript
# job board returns when its body never loaded. Analysing that fragment
# produces a confident, wrong report: skills missing that the posting lists,
# and "Not stated" for requirements plainly written in it.
#
# Below this word count the result is treated as a failed fetch and the person
# is told to paste the text instead. Set well under the length of any genuine
# posting so a short but real advert is never rejected.
MIN_POSTING_WORDS = 80

# Signs that the page served a gate rather than the posting.
_ACCESS_WALL = re.compile(
    r"\b(?:sign in to (?:view|continue|see)|log ?in to (?:view|continue|see)"
    r"|create an account to|please (?:sign|log) ?in|verify you are human"
    r"|enable javascript|access denied|403 forbidden|are you a robot"
    r"|checking your browser)\b",
    re.I,
)

# Containers job boards and ATS platforms (Greenhouse, Lever, Workday, LinkedIn,
# generic career pages, ...) commonly wrap the actual posting in. Tried, in
# order, before ever falling back to the whole page.
_JD_CONTAINER_SELECTORS: list[dict] = [
    {"id": "job-description"},
    {"id": "jobDescriptionText"},
    {"id": "job_description"},
    {"id": re.compile(r"job.?description", re.I)},
    {"class": re.compile(r"job[-_]?description", re.I)},
    {"class": re.compile(r"jobDescription", re.I)},
    {"class": re.compile(r"posting[-_]?(content|description|requirements)", re.I)},
    {"class": re.compile(r"job[-_]?details", re.I)},
    {"class": re.compile(r"description[-_]?content", re.I)},
    {"data-testid": re.compile(r"job.*description", re.I)},
]


def allowed_extension(filename: str, allowed: Iterable[str]) -> bool:
    return Path(filename).suffix.lower() in {item.lower() for item in allowed}


def persist_upload(upload_file: UploadFile, upload_dir: Path) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload_file.filename or "upload").suffix.lower()
    safe_name = f"{Path(upload_file.filename or 'upload').stem}_{uuid.uuid4().hex}{suffix}"
    target_path = upload_dir / safe_name
    content = upload_file.file.read()
    upload_file.file.seek(0)
