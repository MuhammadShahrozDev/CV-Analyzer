"""Batch email generation over a workbook of jobs.

Two responsibilities, deliberately kept apart from each other and from
sending:

  * build the candidate profile ONCE from the uploaded CV
  * loop the job rows, running the existing Phase One analysis per job and
    handing the result to the existing email generator

Nothing here re-implements CV parsing, JD analysis, skill matching or email
writing. It calls the modules that already do those things. The only new logic
is the loop, the error isolation, and keeping the expensive CV work out of it.

Recipient resolution and sending live elsewhere on purpose, so the demo
(everything to one address) can become production (each row to its own
recruiter) without touching generation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .email_generator import (
    build_individual_email_context,
    build_team_email_context,
    generate_email_with_fallback,
)
from .excel_jobs import is_valid_email
from .extractor import extract_skills_from_text
from .matcher import build_match_report
from .parser import clean_extracted_text, extract_text_from_upload, normalize_job_description
from .preprocessing import preprocess_text
from .scorer import build_ats_report

logger = logging.getLogger("ats.batch")

# Gemini's free tier is rate limited per minute. A short pause between rows
# keeps a hundred-row batch from tripping it; without this the tail of the
# batch silently falls back to templates.
DEFAULT_THROTTLE_SECONDS = 0.6

# Beyond this a single description is truncated before analysis. Some postings
# run to 20,000 characters, and the tail is almost always benefits and legal
# boilerplate that slows every stage without changing the result.
MAX_DESCRIPTION_CHARS = 12000


@dataclass
class CandidateProfile:
    """The CV, parsed once and reused for every job in the batch.

    Holds the raw text and the extracted skills, not a rendered email. Keeping
    it separate from the job loop is what will let a future version run
    several candidates against the same workbook.
    """

    filename: str
    layout_text: str
    flat_text: str
    clean_text: str
    skills: dict[str, Any]
    name: str = ""

    @property
    def is_usable(self) -> bool:
        return bool(self.flat_text.strip())


@dataclass
class JobEmailResult:
    """One row's outcome. Always produced, successful or not."""

    row: int
    role: str = ""
    company: str = ""
    recipient: str = ""
    status: str = "pending"      # generated | skipped | failed
    subject: str = ""
    body: str = ""
    source: str = ""             # gemini | template
    error: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "role": self.role,
            "company": self.company,
            "recipient": self.recipient,
            "status": self.status,
            "subject": self.subject,
            "body": self.body,
            "source": self.source,
            "error": self.error,
            **self.extras,
        }


def build_candidate_profile(cv_path: str | Path, skills_config: dict[str, Any]) -> CandidateProfile:
    """Parse the CV once.

    Called a single time per batch. Doing this inside the job loop would
    re-read and re-analyse the same PDF a hundred times for no benefit.
    """
    path = Path(cv_path)
    layout_text = extract_text_from_upload(path, preserve_layout=True)
    flat_text = clean_extracted_text(layout_text)

    if not flat_text.strip():
        raise ValueError("No readable text could be extracted from the CV.")

    return CandidateProfile(
        filename=path.name,
        layout_text=layout_text,
        flat_text=flat_text,
        clean_text=preprocess_text(layout_text),
        skills=extract_skills_from_text(layout_text, skills_config),
    )


def resolve_recipient(
    job: dict[str, Any],
    *,
    demo_recipient: str = "",
    use_row_email: bool = True,
) -> tuple[str, str]:
    """Decide where a row's email goes. Returns (address, reason).

    The only place a recipient is chosen. Demo mode sends everything to one
    configurable address; production reads each row's own contact. Switching
    between them changes nothing about how the email is written.

    No address is defaulted, embedded or hard-coded anywhere - an empty
    demo_recipient with no row email is an error the caller must surface.
    """
    row_email = (job.get("recipient_email") or "").strip()

    if use_row_email and row_email:
        if is_valid_email(row_email):
            return row_email, "row"
        return "", f"The address in this row is not valid: {row_email[:60]}"

    demo = (demo_recipient or "").strip()
    if demo:
        if is_valid_email(demo):
            return demo, "demo"
        return "", f"The recipient address entered is not valid: {demo[:60]}"

    return "", "No recipient: this row has no email address and no recipient was set."


def _analyse_job(profile: CandidateProfile, description: str, skills_config: dict[str, Any]):
    """Run the existing Phase One pipeline for one job description."""
    if len(description) > MAX_DESCRIPTION_CHARS:
        description = description[:MAX_DESCRIPTION_CHARS]

    job_flat, job_layout = normalize_job_description(description)
    job_skills = extract_skills_from_text(job_layout, skills_config)

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
    return build_ats_report(match_data), job_flat, job_layout


def generate_batch(
    jobs: list[dict[str, Any]],
    profile: CandidateProfile,
    skills_config: dict[str, Any],
    *,
    mode: str = "individual",
    demo_recipient: str = "",
    use_row_email: bool = True,
    limit: Optional[int] = None,
    throttle_seconds: float = DEFAULT_THROTTLE_SECONDS,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> dict[str, Any]:
    """Generate one email per job row. Never sends anything.

    A failure on one row is recorded against that row and the batch carries
    on - a single malformed description must not cost the other ninety-nine.
    """
    mode = mode if mode in ("team", "individual") else "individual"
    selected = jobs[:limit] if limit else jobs
    total = len(selected)
    results: list[JobEmailResult] = []

    for position, job in enumerate(selected, start=1):
        result = JobEmailResult(
            row=job.get("row", position),
            role=(job.get("role") or "").strip(),
            company=(job.get("company") or "").strip(),
        )

        if progress:
            progress(position, total, result.role)

        # Rows the reader already flagged (no role, unusable description).
        if job.get("skip_reason"):
            result.status = "skipped"
            result.error = job["skip_reason"]
            results.append(result)
            continue

        recipient, reason = resolve_recipient(
            job, demo_recipient=demo_recipient, use_row_email=use_row_email
        )
        result.recipient = recipient
        if not recipient:
            result.status = "skipped"
            result.error = reason
            results.append(result)
            continue

        try:
            report, job_flat, job_layout = _analyse_job(
                profile, job.get("description", ""), skills_config
            )

            if mode == "team":
                context = build_team_email_context(
                    report=report,
                    job_text=job_flat,
                    job_name=result.role or job.get("url", ""),
                    job_layout_text=job_layout,
                )
            else:
                context = build_individual_email_context(
                    report=report,
                    resume_layout_text=profile.layout_text,
                    job_text=job_flat,
                    resume_filename=profile.filename,
                    job_name=result.role or job.get("url", ""),
                    job_layout_text=job_layout,
                )

            # The workbook's own role and company are more reliable than
            # anything inferred from the description, so they win.
            if result.role:
                context["job_title"] = result.role
            if result.company:
                context["company"] = result.company

            email = generate_email_with_fallback(mode, context)
            result.subject = email.get("subject", "")
            result.body = email.get("body", "")
            result.source = email.get("source", "")
            result.status = "generated"

        except Exception as exc:  # one bad row must not end the batch
            logger.exception("Email generation failed for row %s", result.row)
            result.status = "failed"
            result.error = f"{type(exc).__name__}: {exc}"[:200]

        results.append(result)

        if throttle_seconds and position < total:
            time.sleep(throttle_seconds)

    generated = sum(1 for item in results if item.status == "generated")
    return {
        "mode": mode,
        "total": total,
        "generated": generated,
        "skipped": sum(1 for item in results if item.status == "skipped"),
        "failed": sum(1 for item in results if item.status == "failed"),
        "recipient_mode": "row" if use_row_email else "demo",
        "results": [item.as_dict() for item in results],
    }
