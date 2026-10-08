from __future__ import annotations

import os
import re
import secrets
import smtplib
import ssl
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from utils.batch_processor import CandidateProfile, build_candidate_profile
from utils.email_generator import (
    build_individual_email_context,
    build_team_email_context,
    extract_candidate_name,
    generate_email_with_fallback,
)
from utils.email_sender import SmtpSettings, build_message
from utils.extractor import extract_skills_from_text, load_skills_config
from utils.matcher import build_match_report
from utils.parser import (
    allowed_extension,
    normalize_job_description,
    persist_upload,
)
from utils.preprocessing import preprocess_text
from utils.scorer import build_ats_report


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
SKILLS_FILE = BASE_DIR / "skills.json"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

RESUME_EXTENSIONS = {".pdf", ".docx"}

skills_config = load_skills_config(SKILLS_FILE)


router = APIRouter(
    prefix="/api/v1",
    tags=["SaaS API"],
)


# =========================================================
# AUTHENTICATION
# =========================================================

def require_api_token(
    authorization: Optional[str],
) -> None:
    expected_token = (
        os.environ.get("SAAS_API_TOKEN", "")
        .strip()
    )

    if not expected_token:
        raise HTTPException(
            status_code=503,
            detail="SAAS_API_TOKEN is not configured on the service.",
        )

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header.",
        )

    prefix = "Bearer "

    if not authorization.startswith(prefix):
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header.",
        )

    supplied_token = authorization[len(prefix):].strip()

    if not supplied_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid API token.",
        )

    if not secrets.compare_digest(
        supplied_token,
        expected_token,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API token.",
        )


def api_auth_dependency(
    authorization: Optional[str] = Header(default=None),
) -> None:
    require_api_token(authorization)


protected_router = APIRouter(
    dependencies=[Depends(api_auth_dependency)]
)


# =========================================================
# REQUEST MODELS
# =========================================================

class CandidateProfilePayload(BaseModel):
    filename: str = ""
    layout_text: str = ""
    flat_text: str = ""
    clean_text: str = ""
    skills: dict[str, Any] = Field(default_factory=dict)
    name: str = ""
    email: str = ""
    phone: str = ""
    summary: str = ""
    experience: list[dict[str, Any]] = Field(default_factory=list)
    education: list[dict[str, Any]] = Field(default_factory=list)


class JobPayload(BaseModel):
    id: Optional[int] = None
    title: str = ""
    company: str = ""
    description: str
    url: str = ""
    recipient_name: str = ""
    recipient_email: str = ""


class MatchRequest(BaseModel):
    profile: CandidateProfilePayload
    job: JobPayload


class EmailGenerateRequest(BaseModel):
    profile: CandidateProfilePayload
    job: JobPayload
    mode: str = "individual"
    vary: bool = False


class SMTPPayload(BaseModel):
    host: str
    port: int = 587
    username: str
    password: str
    from_address: str
    from_name: str = ""
    encryption: str = "tls"


class EmailSendItem(BaseModel):
    row: Optional[int] = None
    recipient: str
    subject: str
    body: str
    role: str = ""
    company: str = ""
    reply_to: str = ""
    attachment_path: str = ""
    attachment_name: str = ""


class EmailSendRequest(BaseModel):
    smtp: SMTPPayload
    emails: list[EmailSendItem]
    dry_run: bool = False
    delay_seconds: float = 2.0


# =========================================================
# HELPERS
# =========================================================

def candidate_from_payload(
    payload: CandidateProfilePayload,
) -> CandidateProfile:
    return CandidateProfile(
        filename=payload.filename,
        layout_text=payload.layout_text,
        flat_text=payload.flat_text,
        clean_text=payload.clean_text,
        skills=payload.skills,
        name=payload.name,
    )



SECTION_ALIASES = {
    "summary": {"summary", "professional summary", "profile", "career summary", "objective", "career objective", "about me"},
    "experience": {"experience", "work experience", "professional experience", "employment history", "work history", "career history", "employment"},
    "education": {"education", "academic background", "academic qualifications", "qualifications", "education & qualifications", "education and qualifications"},
    "skills": {"skills", "technical skills", "core skills", "key skills", "competencies", "technical expertise"},
    "projects": {"projects", "project experience", "academic projects", "personal projects"},
    "certifications": {"certifications", "certificates", "licenses & certifications", "licenses and certifications", "courses"},
}

ALL_SECTION_HEADERS = {
    alias: key
    for key, aliases in SECTION_ALIASES.items()
    for alias in aliases
}

MONTHS = (
    "jan|january|feb|february|mar|march|apr|april|may|jun|june|"
    "jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december"
)

DATE_RANGE_RE = re.compile(
    rf"(?i)\b(?:(?:{MONTHS})\s+)?((?:19|20)\d{{2}})\s*"
    rf"(?:-|–|—|to)\s*"
    rf"(?:(?:(?:{MONTHS})\s+)?((?:19|20)\d{{2}})|present|current|now)\b"
)

YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
EMAIL_RE = re.compile(r"(?i)(?<![\w.+-])([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})(?![\w.-])")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s().-]*)?(?:\d[\s().-]*){9,14}(?!\d)")

DEGREE_WORDS = (
    "bachelor", "master", "phd", "doctorate", "bsc", "bs ", "bs(", "b.s", "msc", "ms ", "m.s", "mba",
    "bba", "ba ", "b.a", "ma ", "m.a", "associate", "diploma", "degree", "intermediate", "matric", "secondary"
)

ROLE_WORDS = (
    "engineer", "developer", "manager", "analyst", "consultant", "specialist", "officer", "executive", "lead",
    "intern", "designer", "administrator", "coordinator", "scientist", "architect", "supervisor", "director",
    "accountant", "teacher", "lecturer", "researcher", "assistant", "associate", "technician", "sales", "marketing"
)


def _clean_line(value: str) -> str:
    value = re.sub(r"[\t ]+", " ", value or "")
    return value.strip(" \t|•·▪◦")


def _lines(text: str) -> list[str]:
    return [_clean_line(line) for line in (text or "").replace("\r", "\n").split("\n")]


def _header_key(line: str) -> Optional[str]:
    normalized = re.sub(r"[^a-z0-9& ]+", "", (line or "").lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    if len(normalized) > 40:
        return None
    return ALL_SECTION_HEADERS.get(normalized)


def _section(text: str, wanted: str) -> list[str]:
    lines = _lines(text)
    collecting = False
    result: list[str] = []

    for line in lines:
        key = _header_key(line)
        if key:
            if collecting and key != wanted:
                break
            collecting = key == wanted
            continue
        if collecting:
            result.append(line)

    while result and not result[0]:
        result.pop(0)
    while result and not result[-1]:
        result.pop()
    return result


def _extract_email(text: str) -> str:
    match = EMAIL_RE.search(text or "")
    return match.group(1).strip() if match else ""


def _extract_phone(text: str) -> str:
    for raw in PHONE_RE.findall(text or ""):
        candidate = re.sub(r"\s+", " ", raw).strip(" .,-")
        digits = re.sub(r"\D", "", candidate)
        if 10 <= len(digits) <= 15 and not YEAR_RE.fullmatch(digits):
            return candidate
    return ""


def _extract_summary(layout_text: str) -> str:
    summary_lines = [line for line in _section(layout_text, "summary") if line]
    if summary_lines:
        return " ".join(summary_lines[:8])[:2000]

    candidates = []
    for line in _lines(layout_text)[:25]:
        if not line or _header_key(line):
            continue
        if "@" in line or PHONE_RE.search(line) or "linkedin.com" in line.lower() or "github.com" in line.lower():
            continue
        if len(line.split()) >= 8:
            candidates.append(line)
        if len(candidates) >= 3:
            break
    return " ".join(candidates)[:2000]


def _to_date(year: Optional[str]) -> Optional[str]:
    return f"{year}-01-01" if year else None


def _date_info(text: str) -> tuple[Optional[str], Optional[str], bool]:
    match = DATE_RANGE_RE.search(text or "")
    if match:
        start_year = match.group(1)
        raw_end = match.group(2)
        current = bool(re.search(r"(?i)\b(?:present|current|now)\b", match.group(0)))
        end_year = None if current else raw_end
        return _to_date(start_year), _to_date(end_year), current

    years = YEAR_RE.findall(text or "")
    if len(years) >= 2:
        return _to_date(years[0]), _to_date(years[1]), False
    if len(years) == 1:
        return _to_date(years[0]), None, False
    return None, None, False


def _split_blocks(section_lines: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []

    for line in section_lines:
        if not line:
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)

    if current:
        blocks.append(current)

    if len(blocks) <= 1 and section_lines:
        blocks = []
        current = []
        for line in [x for x in section_lines if x]:
            if current and DATE_RANGE_RE.search(line) and any(DATE_RANGE_RE.search(x) for x in current):
                blocks.append(current)
                current = []
            current.append(line)
        if current:
            blocks.append(current)

    return blocks


def _looks_like_role(value: str) -> bool:
    lower = (value or "").lower()
    return any(word in lower for word in ROLE_WORDS)


def _looks_like_degree(value: str) -> bool:
    lower = f" {(value or '').lower()} "
    return any(word in lower for word in DEGREE_WORDS)


def _extract_experience(layout_text: str) -> list[dict[str, Any]]:
    lines = _section(layout_text, "experience")
    if not lines:
        return []

    rows: list[dict[str, Any]] = []

    for index, block in enumerate(_split_blocks(lines)):
        clean = [line.strip() for line in block if line and line.strip()]
        if not clean:
            continue

        joined = " | ".join(clean)
        start_date, end_date, is_current = _date_info(joined)

        title = ""
        company = ""
        location = ""

        first_line = clean[0]
        identity_line = re.sub(DATE_RANGE_RE, "", first_line).strip(" |-–—,")
        parts = [
            part.strip()
            for part in re.split(r"\s*[|•]\s*", identity_line)
            if part.strip()
        ]

        if len(parts) >= 2:
            role_index = next(
                (i for i, part in enumerate(parts) if _looks_like_role(part)),
                None
            )

            if role_index is not None:
                title = parts[role_index]
                remaining = [
                    part for i, part in enumerate(parts)
                    if i != role_index
                ]
                if remaining:
                    company = remaining[0]
                if len(remaining) > 1:
                    location = remaining[-1]
            else:
                company = parts[0]
                title = parts[1]
                if len(parts) > 2:
                    location = parts[2]

        else:
            candidate = identity_line
            if candidate:
                if _looks_like_role(candidate):
                    title = candidate
                else:
                    company = candidate

        if not title or not company:
            for line in clean[1:4]:
                stripped = re.sub(DATE_RANGE_RE, "", line).strip(" |-–—,")
                if not stripped:
                    continue

                line_parts = [
                    part.strip()
                    for part in re.split(r"\s*[|•]\s*", stripped)
                    if part.strip()
                ]

                for part in line_parts:
                    if not title and _looks_like_role(part):
                        title = part
                    elif not company and not _looks_like_role(part):
                        company = part
                    elif not location and part not in {title, company} and len(part) < 100:
                        location = part

                if title and company:
                    break

        description_lines = []
        for i, line in enumerate(clean):
            if i == 0:
                continue
            if DATE_RANGE_RE.search(line) and len(line.split()) <= 8:
                continue
            description_lines.append(line)

        if not any([title, company, start_date, description_lines]):
            continue

        rows.append({
            "company_name": company[:255] or None,
            "job_title": title[:255] or None,
            "location": location[:255] or None,
            "start_date": start_date,
            "end_date": end_date,
            "is_current": is_current,
            "description": " ".join(description_lines)[:4000] or None,
            "skills": [],
            "sort_order": index,
        })

    return rows[:20]


def _extract_education(layout_text: str) -> list[dict[str, Any]]:
    lines = _section(layout_text, "education")
    if not lines:
        return []

    rows: list[dict[str, Any]] = []

    for index, block in enumerate(_split_blocks(lines)):
        clean = [line.strip() for line in block if line and line.strip()]
        if not clean:
            continue

        joined = " | ".join(clean)
        start_date, end_date, _ = _date_info(joined)

        degree = ""
        institution = ""
        field = ""

        for line in clean:
            stripped = re.sub(DATE_RANGE_RE, "", line).strip(" |-–—,")
            if not stripped:
                continue

            if not degree and _looks_like_degree(stripped):
                degree = stripped
                continue

            lower = stripped.lower()
            if (
                not institution
                and any(
                    word in lower
                    for word in (
                        "university",
                        "college",
                        "institute",
                        "school",
                        "academy",
                    )
                )
            ):
                institution = stripped

        if degree:
            match = re.search(r"(?i)\b(?:in|of)\s+(.+)$", degree)
            if match:
                field = match.group(1).strip(" ,.-")[:255]
            else:
                field_match = re.search(
                    r"(?i)\b(?:b\.?sc\.?|b\.?s\.?|m\.?sc\.?|m\.?s\.?|mba|bba|ba|ma)\b[ .:-]*(.+)$",
                    degree,
                )
                if field_match:
                    candidate_field = field_match.group(1).strip(" ,.-")
                    if candidate_field:
                        field = candidate_field[:255]

        if not institution:
            for line in clean:
                stripped = re.sub(DATE_RANGE_RE, "", line).strip(" |-–—,")
                if (
                    stripped
                    and stripped != degree
                    and not YEAR_RE.fullmatch(stripped)
                ):
                    institution = stripped
                    break

        description_lines = []
        for line in clean:
            stripped = re.sub(DATE_RANGE_RE, "", line).strip(" |-–—,")
            if not stripped or stripped in {degree, institution}:
                continue
            description_lines.append(stripped)

        if not any([institution, degree, start_date, end_date]):
            continue

        rows.append({
            "institution": institution[:255] or None,
            "degree": degree[:255] or None,
            "field_of_study": field or None,
            "start_date": start_date,
            "end_date": end_date,
            "description": " ".join(description_lines)[:3000] or None,
            "sort_order": index,
        })

    return rows[:15]


def extract_structured_profile(profile: CandidateProfile, filename: str) -> dict[str, Any]:
    layout = profile.layout_text or ""
    candidate_name = extract_candidate_name(layout, filename)
    profile.name = candidate_name

    return {
        "filename": profile.filename,
        "name": profile.name,
        "email": _extract_email(layout),
        "phone": _extract_phone(layout),
        "summary": _extract_summary(layout),
        "skills": profile.skills,
        "experience": _extract_experience(layout),
        "education": _extract_education(layout),
        "layout_text": profile.layout_text,
        "flat_text": profile.flat_text,
        "clean_text": profile.clean_text,
    }


def analyze_job(
    profile: CandidateProfile,
    job: JobPayload,
) -> dict[str, Any]:
    description = job.description or ""

    if not description.strip():
        raise HTTPException(
            status_code=400,
            detail="Job description cannot be empty.",
        )

    if len(description) > 12000:
        description = description[:12000]

    job_flat, job_layout = normalize_job_description(description)

    job_skills = extract_skills_from_text(
        job_layout,
        skills_config,
    )

    match_data = build_match_report(
        resume_text=profile.flat_text,
        job_text=job_flat,
        resume_clean=profile.clean_text,
        job_clean=preprocess_text(job_layout),
        resume_skills=profile.skills,
        job_skills=job_skills,
        skills_config=skills_config,
        resume_layout_text=profile.layout_text,
        job_layout_text=job_layout,
    )

    report = build_ats_report(match_data)

    return {
        "report": report,
        "job_flat": job_flat,
        "job_layout": job_layout,
    }


def smtp_settings_from_payload(
    payload: SMTPPayload,
) -> SmtpSettings:
    encryption = (payload.encryption or "tls").lower()

    use_ssl = (
        encryption == "ssl"
        or payload.port == 465
    )

    use_tls = (
        encryption == "tls"
        and not use_ssl
    )

    return SmtpSettings(
        host=payload.host.strip(),
        port=payload.port,
        username=payload.username.strip(),
        password=payload.password,
        from_address=(
            payload.from_address.strip()
            or payload.username.strip()
        ),
        from_name=payload.from_name.strip(),
        use_ssl=use_ssl,
        use_tls=use_tls,
    )


def smtp_connect(
    settings: SmtpSettings,
) -> smtplib.SMTP:
    context = ssl.create_default_context()

    if settings.use_ssl:
        server = smtplib.SMTP_SSL(
            settings.host,
            settings.port,
            timeout=30,
            context=context,
        )
    else:
        server = smtplib.SMTP(
            settings.host,
            settings.port,
            timeout=30,
        )

        server.ehlo()

        if settings.use_tls:
            server.starttls(context=context)
            server.ehlo()

    server.login(
        settings.username,
        settings.password,
    )

    return server


# =========================================================
# PUBLIC HEALTH
# =========================================================

@router.get("/health")
def api_health():
    return {
        "ok": True,
        "service": "cv-analyzer",
        "version": "1.2.0",
    }


# =========================================================
# PROFILE / CV ANALYSIS
# =========================================================

@protected_router.post("/profile/analyze")
async def profile_analyze(
    resume_file: UploadFile = File(...),
):
    if not resume_file.filename:
        raise HTTPException(
            status_code=400,
            detail="Upload a CV.",
        )

    if not allowed_extension(
        resume_file.filename,
        RESUME_EXTENSIONS,
    ):
        raise HTTPException(
            status_code=400,
            detail="CV must be PDF or DOCX.",
        )

    resume_path = persist_upload(
        resume_file,
        UPLOAD_DIR,
    )

    try:
        profile = build_candidate_profile(
            resume_path,
            skills_config,
        )

        structured = extract_structured_profile(
            profile,
            resume_file.filename,
        )

        return {
            "ok": True,
            "analyzer_version": "1.2.0",
            "profile": structured,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"CV analysis failed: {type(exc).__name__}",
        ) from exc


# =========================================================
# JOB MATCHING
# =========================================================

@protected_router.post("/match")
def match_job(
    request: MatchRequest,
):
    profile = candidate_from_payload(request.profile)

    if not profile.is_usable:
        raise HTTPException(
            status_code=400,
            detail="Candidate profile contains no usable CV text.",
        )

    analysis = analyze_job(
        profile,
        request.job,
    )

    return {
        "ok": True,
        "job_id": request.job.id,
        "report": analysis["report"],
    }


# =========================================================
# EMAIL / PROPOSAL GENERATION
# =========================================================

@protected_router.post("/email/generate")
def generate_email(
    request: EmailGenerateRequest,
):
    profile = candidate_from_payload(request.profile)

    if not profile.is_usable:
        raise HTTPException(
            status_code=400,
            detail="Candidate profile contains no usable CV text.",
        )

    analysis = analyze_job(
        profile,
        request.job,
    )

    report = analysis["report"]
    job_flat = analysis["job_flat"]
    job_layout = analysis["job_layout"]

    mode = (
        request.mode
        if request.mode in ("individual", "team")
        else "individual"
    )

    job_name = (
        request.job.title
        or request.job.url
        or "Job Opportunity"
    )

    if mode == "team":
        context = build_team_email_context(
            report=report,
            job_text=job_flat,
            job_name=job_name,
            job_layout_text=job_layout,
        )
    else:
        context = build_individual_email_context(
            report=report,
            resume_layout_text=profile.layout_text,
            job_text=job_flat,
            resume_filename=profile.filename,
            job_name=job_name,
            job_layout_text=job_layout,
        )

    if request.job.title:
        context["job_title"] = request.job.title

    if request.job.company:
        context["company"] = request.job.company

    result = generate_email_with_fallback(
        mode,
        context,
        force_llm_variation=request.vary,
    )

    return {
        "ok": True,
        "job_id": request.job.id,
        "mode": mode,
        "subject": result.get("subject", ""),
        "body": result.get("body", ""),
        "source": result.get("source", ""),
        "context": context,
        "match_report": report,
    }


# =========================================================
# EMAIL SENDING
# =========================================================

@protected_router.post("/email/send")
def send_email(
    request: EmailSendRequest,
):
    if not request.emails:
        raise HTTPException(
            status_code=400,
            detail="No emails supplied.",
        )

    if len(request.emails) > 500:
        raise HTTPException(
            status_code=400,
            detail="Maximum 500 emails per request.",
        )

    settings = smtp_settings_from_payload(request.smtp)

    if request.dry_run:
        results = []

        for position, item in enumerate(
            request.emails,
            start=1,
        ):
            results.append(
                {
                    "row": item.row or position,
                    "recipient": item.recipient,
                    "subject": item.subject,
                    "role": item.role,
                    "company": item.company,
                    "status": "dry-run",
                    "error": "",
                }
            )

        return {
            "ok": True,
            "total": len(results),
            "sent": 0,
            "failed": 0,
            "skipped": 0,
            "dry_run": True,
            "results": results,
        }

    server: Optional[smtplib.SMTP] = None
    results: list[dict[str, Any]] = []
    sent_since_connect = 0
    reconnect_every = 20

    try:
        for position, item in enumerate(
            request.emails,
            start=1,
        ):
            outcome = {
                "row": item.row or position,
                "recipient": item.recipient,
                "subject": item.subject,
                "role": item.role,
                "company": item.company,
                "status": "pending",
                "error": "",
            }

            recipient = item.recipient.strip()

            if (
                not recipient
                or "@"
                not in recipient
            ):
                outcome["status"] = "skipped"
                outcome["error"] = "Invalid recipient."
                results.append(outcome)
                continue

            if not item.body.strip():
                outcome["status"] = "skipped"
                outcome["error"] = "Email body is empty."
                results.append(outcome)
                continue

            try:
                if (
                    server is None
                    or sent_since_connect >= reconnect_every
                ):
                    if server is not None:
                        try:
                            server.quit()
                        except Exception:
                            pass

                    server = smtp_connect(settings)
                    sent_since_connect = 0

                message = build_message(
                    to_address=recipient,
                    subject=item.subject,
                    body=item.body,
                    settings=settings,
                    reply_to=item.reply_to,
                    attachment_path=item.attachment_path,
                    attachment_name=item.attachment_name,
                )

                server.send_message(message)

                sent_since_connect += 1
                outcome["status"] = "sent"

            except smtplib.SMTPAuthenticationError:
                outcome["status"] = "failed"
                outcome["error"] = "SMTP authentication failed."
                results.append(outcome)

                for remaining in request.emails[position:]:
                    results.append(
                        {
                            "row": remaining.row,
                            "recipient": remaining.recipient,
                            "subject": remaining.subject,
                            "role": remaining.role,
                            "company": remaining.company,
                            "status": "failed",
                            "error": (
                                "Not attempted because sending stopped "
                                "after SMTP authentication failed."
                            ),
                        }
                    )

                break

            except Exception as exc:
                outcome["status"] = "failed"
                outcome["error"] = (
                    f"{type(exc).__name__}: {exc}"
                )[:300]

                if server is not None:
                    try:
                        server.quit()
                    except Exception:
                        pass

                server = None
                sent_since_connect = 0

            results.append(outcome)

            if (
                request.delay_seconds > 0
                and position < len(request.emails)
            ):
                time.sleep(request.delay_seconds)

    finally:
        if server is not None:
            try:
                server.quit()
            except Exception:
                pass

    return {
        "ok": True,
        "total": len(results),
        "sent": sum(
            1
            for item in results
            if item["status"] == "sent"
        ),
        "failed": sum(
            1
            for item in results
            if item["status"] == "failed"
        ),
        "skipped": sum(
            1
            for item in results
            if item["status"] == "skipped"
        ),
        "dry_run": False,
        "results": results,
    }


# =========================================================
# SMTP TEST
# =========================================================

@protected_router.post("/email/smtp-check")
def smtp_check(
    smtp: SMTPPayload,
):
    settings = smtp_settings_from_payload(smtp)

    try:
        server = smtp_connect(settings)
        server.quit()

        return {
            "ok": True,
            "host": settings.host,
            "port": settings.port,
            "from": settings.from_address,
        }

    except smtplib.SMTPAuthenticationError:
        return {
            "ok": False,
            "error": "SMTP authentication failed.",
        }

    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


router.include_router(protected_router)
