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
    target_path.write_bytes(content)
    return target_path


def extract_text_from_upload(file_path: Path, preserve_layout: bool = False) -> str:
    extension = file_path.suffix.lower()
    if extension == ".pdf":
        return extract_text_from_pdf(file_path, preserve_layout=preserve_layout)
    if extension == ".docx":
        return extract_text_from_docx(file_path, preserve_layout=preserve_layout)
    if extension == ".txt":
        return extract_text_from_txt(file_path, preserve_layout=preserve_layout)
    raise ValueError(f"Unsupported file extension: {extension}")


# ---------------------------------------------------------------------------
# URL extraction
#
# Goal: whatever the resume-vs-job scoring pipeline sees from a URL should be
# the job posting itself, not the surrounding webpage. Previously this always
# rendered the page and returned the entire <body> text, so nav links, cookie
# banners, "related jobs" widgets and footers were fed into skill/keyword
# matching alongside the real requirements - which is why a pasted JD and the
# same JD's URL produced different scores.
#
# The strategy below tries the cheapest, cleanest method first and only pays
# for a headless browser when the page genuinely needs JavaScript to render.
# ---------------------------------------------------------------------------
def _fetch_html(url: str, timeout: float = 15.0) -> str:
    request = Request(url, headers=_REQUEST_HEADERS)
    with urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return raw.decode("utf-8", errors="ignore")


def _extract_with_trafilatura(html: str, url: str = "") -> str:
    """Article/posting-body extraction, stripping nav/ads/footers/boilerplate."""
    if trafilatura is None or not html:
        return ""
    try:
        extracted = trafilatura.extract(
            html,
            url=url or None,
            favor_recall=True,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )
    except Exception:
        extracted = None
    return (extracted or "").strip()


def _extract_main_container_text(html: str) -> str:
    """Look for a known job-description container before falling back further."""
    if BeautifulSoup is None or not html:
        return ""
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    for selector in _JD_CONTAINER_SELECTORS:
        node = soup.find(attrs=selector)
        if node is not None:
            text = node.get_text("\n", strip=True)
            if len(text.split()) >= MIN_USABLE_WORDS:
                return text

    for tag_name in ("main", "article"):
        node = soup.find(tag_name)
        if node is not None:
            text = node.get_text("\n", strip=True)
            if len(text.split()) >= MIN_USABLE_WORDS:
                return text

    return ""


def _render_with_playwright_sync(url: str, timeout: int) -> str:
    """Blocking render, meant to run in a worker thread (see caller below).

    Uses Playwright's synchronous API rather than the async one. The async
    API launches Chromium through asyncio's subprocess transport, which
    depends on the calling thread's event loop supporting subprocesses -
    unreliable on Windows, where uvicorn's --reload supervisor does not
    reliably hand the server process a Proactor loop even when one is
    requested. The sync API manages its driver process with plain blocking
    I/O instead, so it works regardless of what kind of event loop uvicorn
    is using, as long as it runs outside that loop's own thread.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(user_agent=_REQUEST_HEADERS["User-Agent"])
            page.goto(url, wait_until="networkidle", timeout=timeout)
            page.wait_for_timeout(2000)
            html = page.content()
        finally:
            browser.close()
    return html


async def _render_with_playwright(url: str, timeout: int) -> str:
    """Render a JS-heavy page and return the fully-rendered HTML.

    Playwright is optional: importing it lazily here means the rest of the
    app (file uploads, manual paste) works even in environments where
    Playwright or its browser binaries are not installed. The actual render
    runs in a worker thread via the synchronous API (see
    _render_with_playwright_sync) so it does not depend on the request's
    event loop supporting subprocess creation.
    """
    if importlib.util.find_spec("playwright.sync_api") is None:
        # HTTPException rather than RuntimeError: app.py renders the detail of
        # an HTTPException on the upload page, while any other exception is
        # caught by the generic handler and replaced with "something went
        # wrong" - which hid this advice exactly when it was needed.
        raise HTTPException(
            status_code=400,
            detail=(
                "This page builds its content with JavaScript and the optional "
                "browser renderer is not installed, so the description could not "
                "be read. Paste the job description into the text box, or install "
                "the renderer with 'pip install playwright' followed by "
                "'python -m playwright install chromium'."
            ),
        )

    try:
        html = await asyncio.to_thread(_render_with_playwright_sync, url, timeout)
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not launch the headless browser. Stop the server "
                "completely, open a fresh terminal and start it again rather "
                "than relying on --reload. Paste the job description into the "
                "text box in the meantime."
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not render this job page in a headless browser "
                f"({type(exc).__name__}). The Chromium browser binary may not be "
                "installed - run 'python -m playwright install chromium' - or the "
                "page blocked automated access. Paste the job description into the "
                "text box instead."
            ),
        ) from exc

    return html


async def extract_text_from_url(
    url: str,
    timeout: int = 30000,
    preserve_layout: bool = False,
) -> str:
    """Extract job description text from a URL.

    Supports:
    - HTML job pages (static or JavaScript-rendered)
    - Direct PDF, DOCX, and TXT links

    For HTML pages, strategies are tried in order and the first one that
    yields a substantial result wins:
      1. Static fetch -> trafilatura main-content extraction
      2. Static fetch -> known job-description container
      3. Playwright-rendered HTML -> trafilatura main-content extraction
      4. Playwright-rendered HTML -> known job-description container
      5. Playwright-rendered page text (last resort)

    The result is passed through the same cleaning functions used for
    uploaded files and manually pasted text, so downstream scoring treats
    all three input methods consistently.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Invalid URL")

    if parsed.path.lower().endswith(".pdf"):
        request = Request(url, headers=_REQUEST_HEADERS)
        with urlopen(request, timeout=20) as response:
            payload = response.read()
        return extract_text_from_pdf_bytes(payload, preserve_layout=preserve_layout)

    if parsed.path.lower().endswith(".docx"):
        request = Request(url, headers=_REQUEST_HEADERS)
        with urlopen(request, timeout=20) as response:
            payload = response.read()
        return extract_text_from_docx_bytes(payload, preserve_layout=preserve_layout)

    if parsed.path.lower().endswith(".txt"):
        request = Request(url, headers=_REQUEST_HEADERS)
        with urlopen(request, timeout=20) as response:
            text = response.read().decode("utf-8", errors="ignore")
        return clean_extracted_text_preserve_layout(text) if preserve_layout else clean_extracted_text(text)

    candidate = ""
    try:
        static_html = _fetch_html(url)
    except Exception:
        static_html = ""

    if static_html:
        candidate = _extract_with_trafilatura(static_html, url)
        if len(candidate.split()) < MIN_USABLE_WORDS:
            candidate = _extract_main_container_text(static_html)

    if len(candidate.split()) < MIN_USABLE_WORDS:
        # Static fetch was too thin to be the real posting - likely JavaScript
        # rendered. Render it and repeat the same extraction strategy.
        rendered_html = await _render_with_playwright(url, timeout)
        candidate = _extract_with_trafilatura(rendered_html, url)
        if len(candidate.split()) < MIN_USABLE_WORDS:
            candidate = _extract_main_container_text(rendered_html)
        if len(candidate.split()) < MIN_USABLE_WORDS:
            candidate = clean_html_text(rendered_html, preserve_layout=True)

    if not candidate.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "No readable text could be extracted from that link. "
                "Paste the job description into the text box instead."
            ),
        )

    _require_usable_posting(candidate, url)

    return clean_extracted_text_preserve_layout(candidate) if preserve_layout else clean_extracted_text(candidate)


def _require_usable_posting(text: str, url: str) -> None:
    """Reject a fetch that clearly did not return the posting itself.

    Failing loudly here matters more than it looks. A half-loaded page still
    scores, still renders, and still fills every panel - it just does so from
    the wrong text, and nothing on screen says so.
    """
    words = len(text.split())

    if _ACCESS_WALL.search(text[:4000]):
        raise HTTPException(
            status_code=400,
            detail=(
                "That page asked for a sign-in or a browser check instead of showing "
                "the job description, so only the page frame was downloaded. "
                "Open the link in your browser, copy the description, and paste it "
                "into the text box."
            ),
        )

    if words < MIN_POSTING_WORDS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Only {words} words were downloaded from that link - too little to be "
                "the job description. Pages that build their content with JavaScript "
                "(LinkedIn, Work at a Startup, some Greenhouse and Lever embeds) often "
                "need the optional browser renderer: install it with "
                "'pip install playwright' then 'python -m playwright install chromium'. "
                "Otherwise, paste the job description into the text box - it gives the "
                "same result."
            ),
        )


def extract_text_from_pdf(file_path: Path, preserve_layout: bool = False) -> str:
    extracted_parts: list[str] = []

    try:
        reader = PdfReader(str(file_path))

        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""

            if page_text.strip():
                extracted_parts.append(page_text)

    except Exception:
        extracted_parts = []

    if extracted_parts:
        extracted_text = "\n\n".join(extracted_parts)

        return (
            clean_extracted_text_preserve_layout(extracted_text)
            if preserve_layout
            else clean_extracted_text(extracted_text)
        )

    if fitz is not None:
        doc = fitz.open(file_path)

        try:
            extracted_text = "\n\n".join(
                page.get_text() or ""
                for page in doc
            )

            return (
                clean_extracted_text_preserve_layout(extracted_text)
                if preserve_layout
                else clean_extracted_text(extracted_text)
            )

        finally:
            doc.close()

    return ""

def extract_text_from_docx(file_path: Path, preserve_layout: bool = False) -> str:
    document = Document(file_path)
    paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text]
    text = "\n".join(paragraphs)
    return clean_extracted_text_preserve_layout(text) if preserve_layout else clean_extracted_text(text)


def extract_text_from_pdf_bytes(payload: bytes, preserve_layout: bool = False) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_file:
        temp_file.write(payload)
        temp_file.flush()
        return extract_text_from_pdf(Path(temp_file.name), preserve_layout=preserve_layout)


def extract_text_from_docx_bytes(payload: bytes, preserve_layout: bool = False) -> str:
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as temp_file:
        temp_file.write(payload)
        temp_file.flush()
        return extract_text_from_docx(Path(temp_file.name), preserve_layout=preserve_layout)


def extract_text_from_txt(file_path: Path, preserve_layout: bool = False) -> str:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    return clean_extracted_text_preserve_layout(text) if preserve_layout else clean_extracted_text(text)


def clean_extracted_text(text: str) -> str:
    text = re.sub(r"[\r\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_extracted_text_preserve_layout(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.replace("\r", "\n").replace("\t", " ").split("\n")]
    filtered_lines = [line for line in lines if line]
    return "\n".join(filtered_lines).strip()


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._block_breaks = {"p", "div", "section", "article", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table"}
        self._skip_tags = {"script", "style", "noscript", "svg", "nav", "footer"}
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        if tag in self._skip_tags:
            self._skip_depth += 1
            return
        if tag in self._block_breaks:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if data and data.strip():
            self._chunks.append(data)

    def get_text(self, preserve_layout: bool = False) -> str:
        text = " ".join(self._chunks) if not preserve_layout else "".join(self._chunks)
        return clean_extracted_text_preserve_layout(text) if preserve_layout else clean_extracted_text(text)


def clean_html_text(html_text: str, preserve_layout: bool = False) -> str:
    extractor = _HTMLTextExtractor()
    extractor.feed(html_text)
    extractor.close()
    return extractor.get_text(preserve_layout=preserve_layout)


# ---------------------------------------------------------------------------
# Job description normalisation
#
# The same posting reached the scorer in three different shapes depending on
# how it arrived: pasted text was passed through untouched, an uploaded file
# was cleaned, and a URL was cleaned *and* carried whatever navigation the page
# happened to include. Different text means different tokens, which means
# different keyword and similarity scores for what is really one job.
#
# Every path now ends here, so all three inputs produce the same normalised
# document.
# ---------------------------------------------------------------------------

# Lines that belong to the website, not the posting. Matched against a whole
# line so a sentence merely containing one of these words is never dropped.
_WEB_NOISE_LINES = (
    r"cookies?( policy| settings| preferences)?",
    r"(accept|reject|manage|allow)( all)?( cookies| tracking)?",
    r"we use cookies.*",
    r"this (site|website) uses cookies.*",
    r"(please )?enable javascript.*",
    r"skip to (main )?content",
    r"(sign|log) ?(in|up|out)",
    r"create (an )?account",
    r"apply( now| for this job| on company site| with indeed| externally)?",
    r"easy apply",
    r"save( this)? job",
    r"share( this)?( job)?",
    r"report (this )?job",
    r"back to (search|results|jobs|listings)",
    r"(similar|related|recommended|more) jobs.*",
    r"view (all|more).*",
    r"(show|read) more",
    r"see less",
    r"privacy( policy| notice| statement)?",
    r"terms( of (use|service)| and conditions)?",
    r"all rights reserved.*",
    r"copyright.*",
    r"\(c\)\s*\d{4}.*",
    r"\u00a9.*",
    r"follow us.*",
    r"(home|jobs|companies|salaries|about|contact|careers|blog|help|search|menu|login)",
    r"posted \d+ .*ago",
    r"\d+ (applicants?|views?)",
    r"job (id|ref(erence)?|code)\s*[:#].*",
    r"powered by.*",
    r"loading\.*",
)

_WEB_NOISE_PATTERN = re.compile(
    r"^\s*(?:" + "|".join(_WEB_NOISE_LINES) + r")\s*[:.!]?\s*$",
    re.IGNORECASE,
)


def strip_web_noise(text: str) -> str:
    """Drop navigation, cookie banners and site chrome, keeping the posting."""
    kept: list[str] = []
    for line in (text or "").replace("\r", "\n").split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if _WEB_NOISE_PATTERN.match(stripped):
            continue
        kept.append(stripped)
    return "\n".join(kept)


def _drop_consecutive_duplicates(lines: list[str]) -> list[str]:
    """Rendered pages often repeat a heading or button in adjacent nodes."""
    deduped: list[str] = []
    for line in lines:
        if deduped and line.lower() == deduped[-1].lower():
            continue
        deduped.append(line)
    return deduped


def normalize_job_description(text: str) -> tuple[str, str]:
    """Return (flat_text, layout_text) for a job description from any source.

    ``layout_text`` keeps one line per line so section headings can still be
    detected; ``flat_text`` is the single-line form used for keyword work.
    Both are derived from the same cleaned content, which is what makes a
    pasted, uploaded and linked copy of one posting score identically.
    """
    cleaned = strip_web_noise(text)
    lines = _drop_consecutive_duplicates(
        [line for line in clean_extracted_text_preserve_layout(cleaned).split("\n") if line.strip()]
    )
    layout_text = "\n".join(lines).strip()
    flat_text = clean_extracted_text(layout_text)
    return flat_text, layout_text
