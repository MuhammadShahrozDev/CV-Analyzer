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


# =========================================================
# PROFILE EXTRACTION HELPERS
# =========================================================

_PROFILE_SECTION_HEADINGS = {
    "summary", "professional summary", "profile", "professional profile",
    "career summary", "objective", "career objective", "about", "about me",
    "experience", "work experience", "professional experience", "employment",
    "employment history", "work history", "career history",
    "education", "academic background", "academic qualification", "qualifications",
    "skills", "technical skills", "core skills", "competencies", "projects",
    "certifications", "certificates", "awards", "languages", "references",
}


def _profile_lines(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", line).strip()
        for line in (text or "").replace("\r", "\n").split("\n")
    ]


def _heading_key(line: str) -> str:
    value = re.sub(r"[^a-zA-Z ]+", " ", line or "").lower()
    return re.sub(r"\s+", " ", value).strip()


def _extract_email(text: str) -> str:
    match = re.search(
        r"(?i)(?<![\w.+-])([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})(?![\w.-])",
        text or "",
    )
    return match.group(1).strip() if match else ""


def _extract_phone(text: str) -> str:
    candidates = re.findall(
        r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)",
        text or "",
    )
    for raw in candidates:
        digits = re.sub(r"\D", "", raw)
        if 10 <= len(digits) <= 15:
            return re.sub(r"\s+", " ", raw).strip(" .,-")
    return ""


def _extract_section(text: str, wanted: set[str]) -> str:
    raw_lines = (text or "").replace("\r", "\n").split("\n")
    start = None
    output: list[str] = []

    for index, raw in enumerate(raw_lines):
        cleaned = re.sub(r"\s+", " ", raw).strip()
        key = _heading_key(cleaned)

        if start is None:
            if key in wanted:
                start = index + 1
            continue

        if key in _PROFILE_SECTION_HEADINGS and key not in wanted:
            break

        output.append(raw.rstrip())

    return "\n".join(output).strip()


def _extract_summary(text: str) -> str:
    section = _extract_section(
        text,
        {
            "summary", "professional summary", "profile", "professional profile",
            "career summary", "objective", "career objective", "about", "about me",
        },
    )
    if section:
        return re.sub(r"\s+", " ", section).strip()[:2000]
    return ""


def _split_section_blocks(section: str) -> list[str]:
    if not section:
        return []

    blocks = [
        re.sub(r"\n{3,}", "\n\n", block).strip()
        for block in re.split(r"\n\s*\n", section)
        if block.strip()
    ]

    if len(blocks) > 1:
        return blocks[:20]

    # Many CV parsers remove blank lines. In that case, start a new block on a
    # date range line while preserving the preceding identity lines.
    lines = [line for line in section.splitlines() if line.strip()]
    date_re = re.compile(
        r"(?i)\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|"
        r"\d{1,2}[/-])?\s*(?:19|20)\d{2}\b.*(?:-|–|—|to).*"
        r"(?:present|current|now|(?:19|20)\d{2})"
    )

    result: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        current.append(line)
        if date_re.search(line) and len(current) > 1:
            # Keep accumulating description until the next likely identity line.
            continue
        if len(current) >= 8:
            result.append(current)
            current = []
    if current:
        result.append(current)

    return ["\n".join(x).strip() for x in result if x][:20]


def _parse_date_range(text: str) -> tuple[str, str, bool]:
    month = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    date_token = rf"(?:{month}\s+)?((?:19|20)\d{{2}})"
    match = re.search(
        rf"(?i){date_token}\s*(?:-|–|—|to)\s*(?:(?:{month}\s+)?((?:19|20)\d{{2}})|(present|current|now))",
        text or "",
    )
    if not match:
        return "", "", False

    start_year = match.group(1) or ""
    end_year = match.group(2) or ""
    current = bool(match.group(3))
    return (
        f"{start_year}-01-01" if start_year else "",
        "" if current else (f"{end_year}-01-01" if end_year else ""),
        current,
    )


def _extract_experience(text: str) -> list[dict[str, Any]]:
    section = _extract_section(
        text,
        {
            "experience", "work experience", "professional experience", "employment",
            "employment history", "work history", "career history",
        },
    )
    if not section:
        return []

    entries: list[dict[str, Any]] = []
    for position, block in enumerate(_split_section_blocks(section)):
        lines = [re.sub(r"\s+", " ", x).strip() for x in block.splitlines() if x.strip()]
        if not lines:
            continue

        start_date, end_date, is_current = _parse_date_range(block)
        date_line_index = next(
            (i for i, line in enumerate(lines) if re.search(r"\b(?:19|20)\d{2}\b", line)),
            -1,
        )
        identity = lines[:date_line_index] if date_line_index > 0 else lines[:2]
        identity = [x for x in identity if len(x) <= 160]

        job_title = identity[0] if identity else ""
        company = identity[1] if len(identity) > 1 else ""

        entries.append({
            "company_name": company,
            "job_title": job_title,
            "location": "",
            "start_date": start_date,
            "end_date": end_date,
            "is_current": is_current,
            "description": "\n".join(lines)[:5000],
            "skills": [],
            "sort_order": position,
        })

    return entries[:20]


def _extract_education(text: str) -> list[dict[str, Any]]:
    section = _extract_section(
        text,
        {"education", "academic background", "academic qualification", "qualifications"},
    )
    if not section:
        return []

    degree_re = re.compile(
        r"(?i)\b(?:bachelor|master|phd|doctor|b\.?s\.?|b\.?sc\.?|m\.?s\.?|m\.?sc\.?|"
        r"bba|mba|associate|diploma|degree|intermediate|matric)\b"
    )
    school_re = re.compile(r"(?i)\b(?:university|college|institute|school|academy)\b")

    entries: list[dict[str, Any]] = []
    for position, block in enumerate(_split_section_blocks(section)):
        lines = [re.sub(r"\s+", " ", x).strip() for x in block.splitlines() if x.strip()]
        if not lines:
            continue

        institution = next((x for x in lines if school_re.search(x)), "")
        degree = next((x for x in lines if degree_re.search(x)), "")
        start_date, end_date, _ = _parse_date_range(block)

        entries.append({
            "institution": institution,
            "degree": degree,
            "field_of_study": "",
            "start_date": start_date,
            "end_date": end_date,
            "description": "\n".join(lines)[:5000],
            "sort_order": position,
        })

    return entries[:20]


def _profile_summary_payload(profile: CandidateProfile, original_name: str) -> dict[str, Any]:
    layout = profile.layout_text or ""
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
        "source_filename": original_name,
    }


def _nested_number(data: dict[str, Any], *paths: tuple[str, ...]) -> Optional[float]:
    for path in paths:
        value: Any = data
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            if match:
                try:
                    return float(match.group(0))
                except ValueError:
                    pass
    return None


def _recursive_number(data: Any, keys: set[str]) -> Optional[float]:
    if isinstance(data, dict):
        for key, value in data.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            if normalized in keys:
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str):
                    match = re.search(r"-?\d+(?:\.\d+)?", value)
                    if match:
                        try:
                            return float(match.group(0))
                        except ValueError:
                            pass
            found = _recursive_number(value, keys)
            if found is not None:
                return found
    elif isinstance(data, list):
        for value in data:
            found = _recursive_number(value, keys)
            if found is not None:
                return found
    return None


def _normalize_match_report(report: dict[str, Any]) -> dict[str, Any]:
    ats = _nested_number(
        report,
        ("ats_score",), ("overall_score",), ("score",), ("final_score",),
        ("scores", "ats_score"), ("scores", "overall_score"),
    )
    if ats is None:
        ats = _recursive_number(report, {"ats_score", "overall_score", "overall_match", "match_score", "final_score"})

    skill = _nested_number(
        report,
        ("skill_score",), ("technical_score",), ("skills_score",),
        ("scores", "skill_score"), ("scores", "technical_score"),
    )
    if skill is None:
        skill = _recursive_number(report, {"skill_score", "skills_score", "technical_score", "technical_match"})

    keyword = _nested_number(
        report,
        ("keyword_score",), ("keywords_score",),
        ("scores", "keyword_score"),
    )
    if keyword is None:
        keyword = _recursive_number(report, {"keyword_score", "keywords_score", "keyword_match"})

    similarity = _nested_number(
        report,
        ("similarity_score",), ("semantic_similarity",), ("similarity",),
        ("scores", "similarity_score"),
    )
    if similarity is None:
        similarity = _recursive_number(report, {"similarity_score", "semantic_similarity", "similarity", "semantic_score"})

    eligibility = _nested_number(
        report,
        ("eligibility_score",), ("eligibility", "score"),
        ("scores", "eligibility_score"),
    )

    if eligibility is None:
        eligibility = _recursive_number(report, {"eligibility_score", "eligibility_match"})

    available = [x for x in (ats, skill, keyword, similarity, eligibility) if x is not None]
    overall = ats if ats is not None else (sum(available) / len(available) if available else 0.0)

    return {
        "overall_score": round(float(overall), 2),
        "ats_score": None if ats is None else round(ats, 2),
        "skill_score": None if skill is None else round(skill, 2),
        "keyword_score": None if keyword is None else round(keyword, 2),
        "similarity_score": None if similarity is None else round(similarity, 2),
        "eligibility_score": None if eligibility is None else round(eligibility, 2),
        "matched_skills": report.get("matched_skills") or [],
        "missing_skills": report.get("missing_skills") or report.get("missing_technical_skills") or [],
        "matched_keywords": report.get("matched_keywords") or [],
        "missing_keywords": report.get("missing_keywords") or [],
        "reasons": report.get("reasons") or report.get("eligibility") or {},
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
        "version": "1.0.0",
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

        profile.name = extract_candidate_name(
            profile.layout_text,
            resume_file.filename,
        )

        structured_profile = _profile_summary_payload(
            profile,
            resume_file.filename,
        )

        return {
            "ok": True,
            "analyzer_version": "1.1.0",
            "profile": structured_profile,
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

    report = analysis["report"]

    return {
        "ok": True,
        "job_id": request.job.id,
        "scores": _normalize_match_report(report),
        "report": report,
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
