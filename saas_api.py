from __future__ import annotations

import os
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

        candidate_name = extract_candidate_name(
            profile.layout_text,
            resume_file.filename,
        )

        profile.name = candidate_name

        return {
            "ok": True,
            "profile": {
                "filename": profile.filename,
                "name": profile.name,
                "skills": profile.skills,
                "layout_text": profile.layout_text,
                "flat_text": profile.flat_text,
                "clean_text": profile.clean_text,
            },
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
