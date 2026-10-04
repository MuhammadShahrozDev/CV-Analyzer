from __future__ import annotations

import asyncio
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Optional

# On Windows, uvicorn's default event loop (SelectorEventLoop) does not support
# subprocesses. Playwright launches Chromium as a subprocess, so without this
# it fails with "NotImplementedError" even when Chromium is installed
# correctly. ProactorEventLoop supports subprocesses and is the standard fix
# for Playwright + asyncio on Windows. This must be set before uvicorn creates
# its event loop, so it runs here at import time, before anything else.
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from utils.batch_processor import (
    build_candidate_profile,
    generate_batch,
)
from utils.batch_store import store as batch_store
from utils.email_generator import (
    build_individual_email_context,
    build_team_email_context,
    generate_email_with_fallback,
)
from utils.email_sender import (
    SenderNotConfigured,
    is_configured as smtp_is_configured,
    send_batch,
    verify_connection,
)
from utils.excel_jobs import (
    REQUIRED_FIELDS,
    detect_columns,
    inspect_workbook,
    is_valid_email,
    read_jobs,
)
from utils.extractor import extract_skills_from_text, load_skills_config
from utils.matcher import build_match_report
from utils.parser import (
    allowed_extension,
    clean_extracted_text,
    extract_text_from_upload,
    extract_text_from_url,
    normalize_job_description,
    persist_upload,
)
from utils.preprocessing import preprocess_text
from utils.scorer import build_ats_report

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
SKILLS_FILE = BASE_DIR / "skills.json"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("ats.analyzer")

ACCEPTED_RESUME_TYPES = ".pdf, .docx"
ACCEPTED_JOB_TYPES = ".pdf, .docx, .txt"
RESUME_EXTENSIONS = {".pdf", ".docx"}
JOB_EXTENSIONS = {".pdf", ".docx", ".txt"}
DEFAULT_EMAIL_MODE = "individual"

# Batch automation. The workbook may carry any spreadsheet extension; the
# reader validates the contents rather than trusting the name.
WORKBOOK_EXTENSIONS = {".xlsx", ".xlsm"}
ACCEPTED_WORKBOOK_TYPES = ".xlsx, .xlsm"

# Rows per generation request. Small enough that the browser sees steady
# progress on a hundred-row batch, large enough not to spend the whole run on
# HTTP overhead.
BATCH_CHUNK_SIZE = 5
MAX_BATCH_ROWS = 500

app = FastAPI(title="ATS Resume Analyzer", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

skills_config = load_skills_config(SKILLS_FILE)


async def _prepare_job_text(
    job_text_input: str,
    job_url_input: str,
    job_file: Optional[UploadFile],
) -> tuple[str, str, Optional[Path]]:
    """Return (flat_text, layout_text, saved_path) for the job description.

    All three input methods converge on normalize_job_description, so the same
    posting produces the same text - and therefore the same ATS result -
    whether it was pasted, uploaded or linked. Previously pasted text skipped
    cleaning entirely, an upload was parsed twice, and a URL returned its
    layout form in both slots, so the three routes scored differently.
    """
    if job_text_input and job_text_input.strip():
        flat, layout = normalize_job_description(job_text_input)
        if not flat.strip():
            raise HTTPException(status_code=400, detail="The pasted job description is empty.")
        return flat, layout, None

    if job_url_input and job_url_input.strip():
        extracted = await extract_text_from_url(job_url_input.strip(), preserve_layout=True)
        flat, layout = normalize_job_description(extracted)
        if not flat.strip():
            raise HTTPException(status_code=400, detail="Unable to extract text from the job description URL.")
        return flat, layout, None

    if not job_file:
        raise HTTPException(status_code=400, detail="Provide a job description file, URL, or manual text.")

    if not allowed_extension(job_file.filename or "", JOB_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Job description must be PDF, DOCX, or TXT.")

    saved_path = persist_upload(job_file, UPLOAD_DIR)
    # Parsed once. The flat form is derived from the layout form rather than
    # re-opening the document, which halves PDF parsing time per request.
    extracted = extract_text_from_upload(saved_path, preserve_layout=True)
    flat, layout = normalize_job_description(extracted)
    if not flat.strip():
        raise HTTPException(status_code=400, detail="Unable to extract text from the job description file.")
    return flat, layout, saved_path


def _render_index(request: Request, error_message: str = "", status_code: int = 200) -> HTMLResponse:
    """Single place that builds the upload page, with or without an error."""
    context = {
        "request": request,
        "accepted_resume_types": ACCEPTED_RESUME_TYPES,
        "accepted_job_types": ACCEPTED_JOB_TYPES,
    }
    if error_message:
        context["error_message"] = error_message
    html = templates.get_template("index.html").render(context)
    return HTMLResponse(content=html, status_code=status_code)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return _render_index(request)


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    resume_file: UploadFile = File(...),
    job_file: Optional[UploadFile] = File(None),
    job_url: str = Form(""),
    job_text: str = Form(""),
):
    try:
        if not resume_file.filename:
            raise HTTPException(status_code=400, detail="Upload a resume file.")
        if not allowed_extension(resume_file.filename, RESUME_EXTENSIONS):
            raise HTTPException(status_code=400, detail="Resume must be a PDF or DOCX file.")

        resume_path = persist_upload(resume_file, UPLOAD_DIR)
        # Parsed once; the flat form is derived rather than re-parsing the file.
        resume_layout_text = extract_text_from_upload(resume_path, preserve_layout=True)
        resume_text = clean_extracted_text(resume_layout_text)

        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Unable to extract text from the resume.")

        job_text_raw, job_layout_text, job_path = await _prepare_job_text(job_text, job_url, job_file)

        resume_clean = preprocess_text(resume_layout_text)
        job_clean = preprocess_text(job_layout_text)

        resume_skills = extract_skills_from_text(resume_layout_text, skills_config)
        job_skills = extract_skills_from_text(job_layout_text, skills_config)

        match_data = build_match_report(
            resume_text=resume_text,
            job_text=job_text_raw,
            resume_clean=resume_clean,
            job_clean=job_clean,
            resume_skills=resume_skills,
            job_skills=job_skills,
            skills_config=skills_config,
            resume_layout_text=resume_layout_text,
            job_layout_text=job_layout_text,
        )
        ats_report = build_ats_report(match_data)

        job_display_name = (
            job_file.filename if job_file and job_file.filename else (job_url or "Manual Job Description")
        )

        # Phase Two: application email, in two modes. Built entirely from the
        # Phase One report / Logxperts profile above - no separate analysis
        # pipeline. If anything here fails, the ATS results below must still
        # render normally.
        email_contexts: dict[str, dict] = {}
        email_result: Optional[dict] = None
        try:
            email_contexts["individual"] = build_individual_email_context(
                report=ats_report,
                resume_layout_text=resume_layout_text,
                job_text=job_text_raw,
                resume_filename=resume_file.filename,
                job_name=job_display_name,
                # Layout form: the title patterns are line-anchored and cannot
                # match against the flattened text.
                job_layout_text=job_layout_text,
            )
            email_contexts["team"] = build_team_email_context(
                report=ats_report,
                job_text=job_text_raw,
                job_name=job_display_name,
                job_layout_text=job_layout_text,
            )
            email_result = generate_email_with_fallback(
                DEFAULT_EMAIL_MODE, email_contexts[DEFAULT_EMAIL_MODE]
            )
        except Exception:
            logger.exception("Email generation failed; ATS results still rendered.")
            email_result = None

        html = templates.get_template("results.html").render(
            {
                "request": request,
                "report": ats_report,
                "resume_name": resume_file.filename,
                "job_name": job_display_name,
                "resume_path": str(resume_path.name),
                "job_path": str(job_path.name) if job_path else None,
                "email": email_result,
                "email_mode": DEFAULT_EMAIL_MODE,
                "email_contexts_json": json.dumps(email_contexts) if email_contexts else None,
            }
        )
        return HTMLResponse(content=html)
    except HTTPException as exc:
        return _render_index(request, str(exc.detail), exc.status_code)
    except Exception as exc:
        # Full traceback to the server log; a short, non-leaking message to the
        # browser. The previous version echoed repr(exc) to the page, which can
        # expose server paths and library internals to an end user.
        traceback.print_exc()
        logger.exception("Analysis failed for resume=%s", resume_file.filename)
        return _render_index(
            request,
            "Something went wrong while analyzing these documents. "
            f"({type(exc).__name__}) Please try again, or paste the job description as text.",
            500,
        )


@app.post("/regenerate-email")
async def regenerate_email(
    email_type: str = Form(DEFAULT_EMAIL_MODE),
    context: str = Form(...),
    vary: str = Form("false"),
):
    """Regenerates (or switches mode for) the application email.

    `context` is the combined {"individual": {...}, "team": {...}} dict
    built once during /analyze. No re-analysis happens here - only a fresh
    LLM call (or template fallback) against the already-grounded context for
    the requested mode.
    """
    try:
        email_contexts = json.loads(context)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid email context.")

    email_type = email_type if email_type in ("team", "individual") else DEFAULT_EMAIL_MODE
    selected_context = email_contexts.get(email_type)
    if not selected_context:
        raise HTTPException(status_code=400, detail="Missing context for the selected email type.")

    try:
        email_result = generate_email_with_fallback(
            email_type, selected_context, force_llm_variation=(vary == "true")
        )
    except Exception:
        logger.exception("Email regeneration failed.")
        raise HTTPException(status_code=500, detail="Could not regenerate the email. Please try again.")

    email_result["email_type"] = email_type
    return JSONResponse(content=email_result)


# ---------------------------------------------------------------------------
# Batch automation: one CV, many jobs from a workbook.
#
# Three responsibilities kept on separate endpoints, so nothing sends by
# accident and a long run reports progress:
#
#   /batch/inspect   upload CV + workbook, detect columns, open a session
#   /batch/generate  generate ONE chunk of rows; called repeatedly
#   /batch/send      send what was generated, after explicit confirmation
# ---------------------------------------------------------------------------


@app.get("/batch", response_class=HTMLResponse)
def batch_page(request: Request):
    html = templates.get_template("batch.html").render(
        {
            "request": request,
            "accepted_resume_types": ACCEPTED_RESUME_TYPES,
            "accepted_workbook_types": ACCEPTED_WORKBOOK_TYPES,
            "chunk_size": BATCH_CHUNK_SIZE,
            "smtp_ready": smtp_is_configured(),
        }
    )
    return HTMLResponse(content=html)


@app.post("/batch/inspect")
async def batch_inspect(
    resume_file: UploadFile = File(...),
    workbook_file: UploadFile = File(...),
):
    """Parse the CV once, read the workbook, and report the column mapping.

    Nothing is generated here. The response is what the operator confirms
    before any email is written.
    """
    if not resume_file.filename or not allowed_extension(resume_file.filename, RESUME_EXTENSIONS):
        raise HTTPException(status_code=400, detail="The CV must be a PDF or DOCX file.")
    if not workbook_file.filename or not allowed_extension(workbook_file.filename, WORKBOOK_EXTENSIONS):
        raise HTTPException(status_code=400, detail=f"The job file must be {ACCEPTED_WORKBOOK_TYPES}.")

    resume_path = persist_upload(resume_file, UPLOAD_DIR)
    workbook_path = persist_upload(workbook_file, UPLOAD_DIR)

    try:
        # Parsed once for the whole batch - never inside the job loop.
        profile = build_candidate_profile(resume_path, skills_config)
    except Exception as exc:
        logger.exception("CV parsing failed for the batch.")
        raise HTTPException(
            status_code=400,
            detail=f"The CV could not be read ({type(exc).__name__}). Try a different file.",
        ) from exc

    try:
        info = inspect_workbook(workbook_path)
    except Exception as exc:
        logger.exception("Workbook inspection failed.")
        raise HTTPException(
            status_code=400,
            detail=(
                "That file could not be read as a spreadsheet. If it was exported "
                "from a website, open it in Excel and re-save it as .xlsx."
            ),
        ) from exc

    detected = detect_columns(info["columns"])

    if detected["missing_required"]:
        # Refuse rather than generate a hundred emails against the wrong
        # columns. The client shows the picker using `columns` below.
        missing = ", ".join(detected["labels"][field] for field in detected["missing_required"])
        return JSONResponse(
            status_code=200,
            content={
                "ok": False,
                "needs_mapping": True,
                "message": f"Could not identify: {missing}. Choose the correct columns below.",
                "columns": info["columns"],
                "detected": detected,
                "sheet": info["sheet"],
                "header_row": info["header_row"],
                "row_count": info["row_count"],
                "resume_path": resume_path.name,
                "workbook_path": workbook_path.name,
            },
        )

    session = batch_store.create(
        profile=profile,
        workbook_path=str(workbook_path),
        sheet=info["sheet"],
        header_row=info["header_row"],
        mapping=detected["mapping"],
        cv_filename=resume_file.filename,
        workbook_name=workbook_file.filename,
    )
    session.jobs = read_jobs(
        workbook_path, detected["mapping"],
        sheet=info["sheet"], header_row=info["header_row"], limit=MAX_BATCH_ROWS,
    )

    has_row_emails = any((job.get("recipient_email") or "").strip() for job in session.jobs)

    return {
        "ok": True,
        "session_id": session.session_id,
        "sheet": info["sheet"],
        "has_headers": info.get("has_headers", True),
        "row_count": len(session.jobs),
        "columns": info["columns"],
        "detected": detected,
        "has_row_emails": has_row_emails,
        "skipped_rows": sum(1 for job in session.jobs if job.get("skip_reason")),
        "smtp_ready": smtp_is_configured(),
    }


@app.post("/batch/confirm-mapping")
async def batch_confirm_mapping(
    resume_path: str = Form(...),
    workbook_path: str = Form(...),
    sheet: str = Form(""),
    header_row: int = Form(1),
    mapping_json: str = Form(...),
    cv_filename: str = Form(""),
    workbook_name: str = Form(""),
):
    """Open a session using a mapping the operator chose by hand."""
    try:
        mapping = {field: int(index) for field, index in json.loads(mapping_json).items()}
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="The column selection could not be read.")

    for field in REQUIRED_FIELDS:
        if field not in mapping:
            raise HTTPException(status_code=400, detail=f"Choose a column for: {field}.")

    resume = UPLOAD_DIR / Path(resume_path).name
    workbook = UPLOAD_DIR / Path(workbook_path).name
    if not resume.exists() or not workbook.exists():
        raise HTTPException(status_code=400, detail="The uploaded files are no longer available. Upload them again.")

    try:
        profile = build_candidate_profile(resume, skills_config)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="The CV could not be read.") from exc

    session = batch_store.create(
        profile=profile,
        workbook_path=str(workbook),
        sheet=sheet,
        header_row=header_row,
        mapping=mapping,
        cv_filename=cv_filename,
        workbook_name=workbook_name,
    )
    session.jobs = read_jobs(
        workbook, mapping, sheet=sheet or None, header_row=header_row, limit=MAX_BATCH_ROWS
    )

    return {
        "ok": True,
        "session_id": session.session_id,
        "row_count": len(session.jobs),
        "has_row_emails": any((job.get("recipient_email") or "").strip() for job in session.jobs),
        "smtp_ready": smtp_is_configured(),
    }


@app.post("/batch/generate")
async def batch_generate(
    session_id: str = Form(...),
    mode: str = Form(DEFAULT_EMAIL_MODE),
    offset: int = Form(0),
    limit: int = Form(BATCH_CHUNK_SIZE),
    total: int = Form(0),
    demo_recipient: str = Form(""),
    use_row_email: str = Form("false"),
):
    """Generate one chunk. Called repeatedly by the client until done.

    Chunking is what makes a hundred-row run watchable: each call returns in
    seconds and the table fills as it goes, instead of one request hanging for
    several minutes with nothing on screen.
    """
    session = batch_store.get(session_id)
    if not session:
        raise HTTPException(status_code=400, detail="This batch has expired. Upload the files again.")

    use_row = use_row_email == "true"
    recipient = (demo_recipient or "").strip()
    if not use_row and not is_valid_email(recipient):
        raise HTTPException(status_code=400, detail="Enter a valid recipient email address.")

    if offset == 0:
        # A fresh run rather than a continuation.
        session.results = []
        session.mode = mode if mode in ("team", "individual") else DEFAULT_EMAIL_MODE
        session.demo_recipient = recipient
        session.use_row_email = use_row

    wanted = min(total or len(session.jobs), len(session.jobs), MAX_BATCH_ROWS)
    chunk = session.jobs[offset:min(offset + max(1, limit), wanted)]

    if not chunk:
        return {"ok": True, "done": True, "results": [], "summary": session.summary()}

    try:
        outcome = generate_batch(
            chunk,
            session.profile,
            skills_config,
            mode=session.mode,
            demo_recipient=session.demo_recipient,
            use_row_email=session.use_row_email,
        )
    except Exception as exc:
        logger.exception("Batch chunk failed at offset %s", offset)
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed on this chunk ({type(exc).__name__}). Try a smaller number of jobs.",
        ) from exc

    session.results.extend(outcome["results"])
    next_offset = offset + len(chunk)

    return {
        "ok": True,
        "done": next_offset >= wanted,
        "next_offset": next_offset,
        "wanted": wanted,
        "results": outcome["results"],
        "summary": session.summary(),
    }


@app.post("/batch/send")
async def batch_send(
    session_id: str = Form(...),
    confirm: str = Form("false"),
    dry_run: str = Form("false"),
):
    """Send what was generated. Never generates, and never runs unconfirmed."""
    session = batch_store.get(session_id)
    if not session:
        raise HTTPException(status_code=400, detail="This batch has expired. Generate the emails again.")

    if confirm != "true":
        raise HTTPException(status_code=400, detail="Sending was not confirmed.")

    ready = [item for item in session.results if item.get("status") == "generated"]
    if not ready:
        raise HTTPException(status_code=400, detail="There are no generated emails to send.")

    try:
        outcome = send_batch(ready, dry_run=(dry_run == "true"))
    except SenderNotConfigured as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Batch send failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Sending failed ({type(exc).__name__}). Check the server log.",
        ) from exc

    return outcome


@app.get("/batch/smtp-check")
def batch_smtp_check():
    """Confirm the mail account works before a batch is sent."""
    return verify_connection()


@app.get("/health")
def health():
    return {"status": "ok"}