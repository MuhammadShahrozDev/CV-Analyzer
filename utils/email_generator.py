from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger("ats.email")

# ---------------------------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------------------------

# Overridable from .env (GEMINI_MODEL=...) so a model that is not enabled on
# a given key can be swapped without editing this file.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)
GEMINI_TIMEOUT_SECONDS = 25

LOGXPERTS_PROFILE_PATH = Path(__file__).resolve().parent / "logxperts_profile.json"


# ---------------------------------------------------------------------------
# Heuristic extraction: job title / company / candidate name
#
# None of Phase One's modules extract these today, so this is new, additive
# logic. It never touches skill matching, scoring, or eligibility.
# ---------------------------------------------------------------------------

# Labelled patterns. These are the reliable ones - a posting that writes
# "Job Title: X" means it. They are anchored per line, so the caller must pass
# the LAYOUT form of the JD (line structure intact), not the flattened form.
_TITLE_LABELLED = [
    re.compile(r"(?im)^\s*job\s*title\s*[:\-\u2013]\s*(.+)$"),
    re.compile(r"(?im)^\s*position\s*(?:title)?\s*[:\-\u2013]\s*(.+)$"),
    re.compile(r"(?im)^\s*role\s*[:\-\u2013]\s*(.+)$"),
    re.compile(r"(?im)^\s*vacancy\s*[:\-\u2013]\s*(.+)$"),
]

# Prose patterns. Deliberately NOT case-insensitive: the [A-Z] is a real guard
# requiring a capitalised phrase. Under re.IGNORECASE that guard silently did
# nothing, so "We are looking for flexible, proactive engineers" captured the
# adjective "flexible" and it became the subject line.
_TITLE_PROSE = [
    re.compile(r"\b[Ww]e(?:'re| are)\s+(?:looking for|hiring|seeking)\s+(?:an?\s+)?"
               r"([A-Za-z][A-Za-z0-9/&+\-\s]{2,60}?)(?=\s+to\s|\s+who\b|[.,\n]|$)"),
    re.compile(r"\b[Hh]iring\s+(?:an?\s+)?"
               r"([A-Za-z][A-Za-z0-9/&+\-\s]{2,60}?)(?=\s+to\s|\s+who\b|[.,\n]|$)"),
]

# A job title names a role. Requiring one of these words is what separates
# "Full Stack Engineer" from "flexible" or "someone great".
_ROLE_NOUNS = {
    "engineer", "engineering", "developer", "programmer", "architect",
    "analyst", "scientist", "manager", "lead", "director", "consultant",
    "specialist", "administrator", "designer", "researcher", "officer",
    "executive", "associate", "assistant", "coordinator", "intern",
    "internship", "technician", "strategist", "writer", "marketer",
    "accountant", "recruiter", "trainer", "advisor", "supervisor", "head",
    "president", "founder", "cto", "cio", "ceo", "vp", "sre", "devops",
    "designer", "tester", "qa", "scrum", "product", "project", "data",
    "software", "frontend", "backend", "fullstack", "stack", "web", "mobile",
    "cloud", "security", "network", "database", "support", "sales",
    "operations", "hr", "finance", "legal", "content", "graphic",
}

# Fragments that are never a job title, even when they parse like one.
_BAD_TITLE_WORDS = {
    "about", "us", "the", "company", "team", "role", "position", "overview",
    "summary", "description", "requirements", "qualifications", "benefits",
    "responsibilities", "someone", "people", "candidates", "candidate",
    "applicants", "you", "talent", "professionals", "individuals",
}

# Trailing clauses that get swept up by a greedy capture.
_TITLE_TAIL = re.compile(
    r"\s+(?:to\s+\w+|who\s+.*|that\s+.*|with\s+.*|for\s+.*|in\s+order.*)$",
    re.I,
)


def _clean_fragment(value: str, max_words: int = 8) -> str:
    value = re.sub(r"\s+", " ", value).strip(" .,-|\u2013\u2014")
    words = value.split()
    if len(words) > max_words:
        value = " ".join(words[:max_words])
    return value.strip()


def _looks_like_job_title(value: str) -> bool:
    """Whether a captured fragment is plausibly a role name.

    Requires a recognised role noun. Without this the extractor happily
    returned adjectives and sentence fragments, which then appeared verbatim
    in the subject line.
    """
    if not value:
        return False
    words = [word.lower().strip(".,/&-") for word in value.split()]
    if not words or len(words) > 8:
        return False
    if all(word in _BAD_TITLE_WORDS for word in words):
        return False
    if not any(word in _ROLE_NOUNS for word in words):
        return False
    # A real title carries at least one capitalised word. Requiring this keeps
    # all-lowercase prose fragments out while still allowing a leading
    # qualifier, as in "a frontend focused Full Stack Engineer".
    return any(token[:1].isupper() for token in value.split())


def _title_from_filename(job_name: str) -> str:
    """Last resort: job boards often carry the role in the page or file name."""
    if not job_name:
        return ""
    stem = re.sub(r"\.(pdf|docx?|txt)$", "", job_name, flags=re.I)
    stem = re.sub(r"^https?://\S*?/", "", stem)
    stem = re.sub(r"[_\-/]+", " ", stem)
    stem = re.split(r"\bat\b|\|", stem, maxsplit=1)[0]
    stem = _clean_fragment(stem)
    return stem if _looks_like_job_title(stem) else ""


_COMPANY_PATTERNS = [
    re.compile(r"(?im)^\s*company\s*[:\-\u2013]\s*(.+)$"),
    re.compile(r"\bjoin\s+([A-Z][A-Za-z0-9&.,\-]{1,40})\b"),
    re.compile(r"\bat\s+([A-Z][A-Za-z0-9&.\-]{1,40})\s*,?\s+we\b"),
    re.compile(r"\b([A-Z][A-Za-z0-9&.,\-]{1,40})\s+is\s+(?:looking for|seeking|hiring)\b"),
]

# Capitalised words that are technology or section nouns, not employers.
_BAD_COMPANY_WORDS = _BAD_TITLE_WORDS | {
    "our", "your", "this", "a", "an", "apis", "api", "react", "python",
    "typescript", "javascript", "aws", "azure", "gcp", "sql", "docker",
    "kubernetes", "node", "java", "git", "linux", "agile", "scrum",
}


def extract_job_title_company(job_text: str, job_name: str = "") -> dict[str, str]:
    """Best-effort job title and company from the JD.

    Pass the LAYOUT form of the JD. The labelled patterns below are anchored
    per line, and on a flattened single-line string they can only ever match at
    position 0 - which left the weak prose patterns as the only ones that could
    fire, and those were producing adjectives.

    Falls back to the job/file name, then to empty. Never guesses.
    """
    snippet = job_text[:4000] if job_text else ""

    title = ""
    for pattern in _TITLE_LABELLED:
        for match in pattern.finditer(snippet):
            candidate = _clean_fragment(_TITLE_TAIL.sub("", match.group(1)))
            if _looks_like_job_title(candidate):
                title = candidate
                break
        if title:
            break

    if not title:
        for pattern in _TITLE_PROSE:
            for match in pattern.finditer(snippet):
                candidate = _clean_fragment(_TITLE_TAIL.sub("", match.group(1)))
                if _looks_like_job_title(candidate):
                    title = candidate
                    break
            if title:
                break

    # A heading line that is itself the title ("Senior Frontend Engineer").
    if not title:
        for line in snippet.split("\n")[:12]:
            candidate = _clean_fragment(_TITLE_TAIL.sub("", line))
            if 1 < len(candidate.split()) <= 7 and _looks_like_job_title(candidate):
                title = candidate
                break

    if not title:
        title = _title_from_filename(job_name)

    company = ""
    for pattern in _COMPANY_PATTERNS:
        match = pattern.search(snippet)
        if match:
            candidate = _clean_fragment(match.group(1), max_words=6)
            if candidate and candidate.lower() not in _BAD_COMPANY_WORDS:
                company = candidate
                break

    return {"job_title": title, "company": company}


_NAME_TITLECASE_RE = re.compile(r"^[A-Z][a-zA-Z.'\-]+(?:\s+[A-Z][a-zA-Z.'\-]+){1,3}$")
_NAME_ALLCAPS_RE = re.compile(r"^[A-Z][A-Z.'\-]+(?:\s+[A-Z][A-Z.'\-]+){1,3}$")
_NAME_EXCLUDE_WORDS = {
    "resume", "curriculum", "vitae", "cv", "profile", "summary", "objective",
    "contact", "email", "phone", "address", "linkedin", "github", "portfolio",
    "references", "skills", "experience", "education", "projects",
}


def _looks_like_name(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 60:
        return False
    if any(char.isdigit() for char in line):
        return False
    if "@" in line or "http" in line.lower():
        return False
    lowered = line.lower()
    if any(word in lowered for word in _NAME_EXCLUDE_WORDS):
        return False
    if _NAME_TITLECASE_RE.match(line):
        return True
    if _NAME_ALLCAPS_RE.match(line) and len(line.split()) <= 4:
        return True
    return False


def _name_from_filename(filename: str) -> str:
    stem = re.sub(r"\.\w+$", "", filename or "")
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(
        r"(?i)\b(resume|cv|final|updated|copy|draft|latest|v\d+)\b", "", stem
    )
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem.title() if stem else ""


def extract_candidate_name(resume_layout_text: str, resume_filename: str = "") -> str:
    lines = (resume_layout_text or "").replace("\r", "\n").split("\n")
    checked = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        checked += 1
        if checked > 12:
            break
        if _looks_like_name(stripped):
            return stripped

    from_filename = _name_from_filename(resume_filename)
    return from_filename or "the candidate"


# ---------------------------------------------------------------------------
# Individual mode: context assembly from the existing Phase One report only.
# ---------------------------------------------------------------------------


def _role_overview(report: dict[str, Any], job_text: str) -> dict[str, Any]:
    """What the posting is asking for - the material for paragraph one.

    Read entirely from the Phase One report; nothing is re-parsed here.
    """
    breakdown = report.get("responsibility_breakdown", {}) or {}
    duties = [
        item.get("source", "").strip()
        for group in ("Strong Match", "Moderate Match", "Weak Match", "Missing")
        for item in breakdown.get(group, [])
        if item.get("source")
    ]
    eligibility = report.get("eligibility", {}) or {}
    return {
        "required_technical_skills": (report.get("required_technical_skills") or [])[:10],
        "required_soft_skills": (report.get("required_soft_skills") or [])[:5],
        "key_responsibilities": duties[:5],
        "experience_required": eligibility.get("experience_analysis", {}).get("required_label", "")
        or next(
            (c.get("job_value", "") for c in eligibility.get("checks", []) if c.get("key") == "experience"),
            "",
        ),
        "work_mode": next(
            (c.get("job_value", "") for c in eligibility.get("checks", []) if c.get("key") == "work_mode"),
            "",
        ),
        "job_location": next(
            (c.get("job_value", "") for c in eligibility.get("checks", []) if c.get("key") == "location"),
            "",
        ),
    }


def rank_skills_for_role(candidate_skills, role: dict[str, Any]) -> list[str]:
    """The candidate's skills, most relevant to THIS posting first.

    The email should lead with the skills the role actually asks for. Listing
    them alphabetically buried the important ones and made paragraph three read
    like a keyword dump.
    """
    required = [skill.lower() for skill in (role.get("required_technical_skills") or [])]
    required_index = {skill: position for position, skill in enumerate(required)}

    def rank(skill: str) -> tuple[int, int, str]:
        lowered = skill.lower()
        if lowered in required_index:
            return (0, required_index[lowered], lowered)
        # Partial overlap: "Deep Learning" against a required "Machine Learning".
        for position, wanted in enumerate(required):
            if lowered in wanted or wanted in lowered:
                return (1, position, lowered)
        return (2, 0, lowered)

    return sorted({skill for skill in candidate_skills if skill}, key=rank)


def build_individual_email_context(
    *,
    report: dict[str, Any],
    resume_layout_text: str,
    job_text: str,
    resume_filename: str,
    job_name: str,
    job_layout_text: str = "",
) -> dict[str, Any]:
    # Title extraction needs the layout form; job_text may be flattened.
    identity = extract_job_title_company(job_layout_text or job_text, job_name)
    candidate_name = extract_candidate_name(resume_layout_text, resume_filename)

    eligibility = report.get("eligibility", {}) or {}
    checks_by_key = {check.get("key"): check for check in eligibility.get("checks", [])}

    def _check(key: str) -> dict[str, str]:
        check = checks_by_key.get(key, {})
        return {
            "status": check.get("status", "unknown"),
            "detail": check.get("detail", ""),
            "candidate_value": check.get("candidate_value", ""),
            "job_value": check.get("job_value", ""),
        }

    breakdown = report.get("responsibility_breakdown", {}) or {}
    strong_points = [
        item.get("source", "") for item in breakdown.get("Strong Match", [])[:3] if item.get("source")
    ]
    # The resume bullet each requirement matched against - the candidate's own
    # words, which is what paragraph two should be built from.
    candidate_evidence = [
        item.get("target", "")
        for group in ("Strong Match", "Moderate Match")
        for item in breakdown.get(group, [])[:3]
        if item.get("target")
    ]
    moderate_points = [
        item.get("source", "") for item in breakdown.get("Moderate Match", [])[:2] if item.get("source")
    ]

    return {
        "mode": "individual",
        "candidate_name": candidate_name,
        "job_title": identity["job_title"],
        "company": identity["company"],
        "job_name": job_name or "",
        # Paragraph 1 material: what the role itself asks for.
        "role": _role_overview(report, job_text),
        # Paragraph 3 material: the candidate's skills ordered by what this
        # posting asks for, so the email leads with what matters here.
        "relevant_skills": rank_skills_for_role(
            report.get("matched_skills") or [], _role_overview(report, job_text)
        )[:8],
        "candidate_tech_stack": {
            category: values[:6]
            for category, values in (report.get("matched_categories") or {}).items()
            if values
        },
        "matched_skills": (report.get("matched_skills") or [])[:8],
        "partial_skills": (report.get("partial_skills") or [])[:4],
        "candidate_evidence": candidate_evidence[:3],
        "strong_responsibility_matches": strong_points,
        "moderate_responsibility_matches": moderate_points,
        "experience": _check("experience"),
        "location": _check("location"),
        "work_mode": _check("work_mode"),
        "education": _check("education"),
    }


# ---------------------------------------------------------------------------
# Team / B2B mode: JD-driven relevance selection over the Logxperts profile.
# JOB DESCRIPTION -> KEY REQUIREMENTS -> RELEVANCE MATCHING ->
# SELECT RELEVANT EVIDENCE -> GENERATE EMAIL
# The candidate's CV / Phase One results are never touched by this path.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_logxperts_profile() -> dict[str, Any]:
    """Load the Logxperts profile from utils/ or the project root.

    Checking both places means the file can sit in either location without
    Team mode failing outright, which previously took the whole email block
    down with it.
    """
    for path in (
        LOGXPERTS_PROFILE_PATH,
        LOGXPERTS_PROFILE_PATH.parent.parent / "logxperts_profile.json",
    ):
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    return {}


def _normalize_kw(text: str) -> set[str]:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return {token for token in cleaned.split() if len(token) > 1}


def _overlap_score(query_tokens: set[str], keywords: list[str]) -> int:
    candidate_tokens: set[str] = set()
    for kw in keywords:
        candidate_tokens |= _normalize_kw(kw)
    return len(query_tokens & candidate_tokens)


def _jd_keyword_pool(report: dict[str, Any]) -> set[str]:
    """Key requirements pulled from Phase One's own extraction - no new NLP."""
    pool: list[str] = []
    pool.extend(report.get("required_technical_skills") or [])
    pool.extend(report.get("required_soft_skills") or [])
    pool.extend(report.get("context_skills") or [])
    pool.extend(report.get("matched_skills") or [])
    tokens: set[str] = set()
    for item in pool:
        tokens |= _normalize_kw(item)
    return tokens


_DEFAULT_CAPABILITIES = {"Full-Stack Software Development", "Technology Staff Augmentation"}
_DEFAULT_TECH_CATEGORIES = {"Frontend", "Backend"}


def select_relevant_logxperts_evidence(
    jd_tokens: set[str], profile: dict[str, Any], top_capabilities: int = 3, top_projects: int = 2
) -> dict[str, Any]:
    scored_capabilities = [
        (cap, _overlap_score(jd_tokens, cap.get("keywords", [])))
        for cap in profile.get("capability_areas", [])
    ]
    scored_capabilities.sort(key=lambda pair: pair[1], reverse=True)
    relevant_capabilities = [cap for cap, score in scored_capabilities if score > 0][:top_capabilities]
    if not relevant_capabilities:
        relevant_capabilities = [
            cap for cap in profile.get("capability_areas", []) if cap["name"] in _DEFAULT_CAPABILITIES
        ]

    scored_tech: list[tuple[str, dict[str, Any], int]] = []
    for category, details in profile.get("tech_stack", {}).items():
        score = _overlap_score(jd_tokens, details.get("keywords", []))
        scored_tech.append((category, details, score))
    scored_tech.sort(key=lambda triple: triple[2], reverse=True)
    relevant_tech = {cat: details["items"] for cat, details, score in scored_tech if score > 0}
    if not relevant_tech:
        relevant_tech = {
            cat: profile["tech_stack"][cat]["items"]
            for cat in _DEFAULT_TECH_CATEGORIES
            if cat in profile["tech_stack"]
        }

    scored_projects = [
        (proj, _overlap_score(jd_tokens, proj.get("keywords", [])))
        for proj in profile.get("reference_projects", [])
    ]
    scored_projects.sort(key=lambda pair: pair[1], reverse=True)
    relevant_projects = [proj for proj, score in scored_projects if score > 0][:top_projects]

    return {
        "capabilities": [
            {"name": cap["name"], "description": cap["description"]} for cap in relevant_capabilities
        ],
        "tech": relevant_tech,
        "projects": [
            {"name": proj["name"], "summary": proj["summary"]} for proj in relevant_projects
        ],
    }


def build_team_email_context(
    *,
    report: dict[str, Any],
    job_text: str,
    job_name: str,
    job_layout_text: str = "",
) -> dict[str, Any]:
    identity = extract_job_title_company(job_layout_text or job_text, job_name)
    profile = load_logxperts_profile()
    jd_tokens = _jd_keyword_pool(report)
    evidence = select_relevant_logxperts_evidence(jd_tokens, profile)

    # Paragraph 3 material. "Partner details" here means how Logxperts engages -
    # engagement models, delivery method and differentiators. The source
    # document contains no vendor partnerships or certifications, so none are
    # claimed.
    engagement_models = profile.get("engagement_models", []) or []
    scored_models = sorted(
        engagement_models,
        key=lambda model: _overlap_score(jd_tokens, model.get("keywords", [])),
        reverse=True,
    )
    return {
        "mode": "team",
        "job_title": identity["job_title"],
        "company": identity["company"],
        "job_name": job_name or "",
        "role": _role_overview(report, job_text),
        "organization_name": profile.get("organization_name", "Logxperts"),
        "logxperts_engagement_models": scored_models[:3],
        "logxperts_delivery_model": profile.get("delivery_model", [])[:8],
        "logxperts_differentiators": profile.get("differentiators", [])[:3],
        "logxperts_markets": profile.get("markets", []),
        "jd_excerpt": (job_text or "")[:1200],
        "jd_key_requirements": sorted((report.get("required_technical_skills") or [])[:12]),
        "logxperts_capabilities": evidence["capabilities"],
        "logxperts_tech": evidence["tech"],
        "logxperts_projects": evidence["projects"],
    }


# ---------------------------------------------------------------------------
# Deterministic fallbacks - zero fabrication risk by construction. Used when
# no API key is set, the call fails, or the response can't be parsed.
# ---------------------------------------------------------------------------


def _fallback_subject_individual(context: dict[str, Any]) -> str:
    title = context.get("job_title") or ""
    company = context.get("company") or ""
    # "Application for Computer Vision Engineer Role" reads like a real
    # subject line; "Job Application" is what an unparsed JD used to produce.
    if title:
        suffix = "" if re.search(r"\b(role|position|engineer|developer|analyst|manager|"
                                 r"scientist|designer|lead|architect|specialist)\b", title, re.I) else " Role"
        subject = f"Application for {title}{suffix}"
        return f"{subject} at {company}" if company else subject
    strongest = (context.get("relevant_skills") or context.get("matched_skills") or [])
    if strongest:
        return f"Application - {strongest[0]} Role"
    return "Job Application"


# Words a sentence must never end on. Cutting at a word boundary still left
# fragments like "...video analytics and." or "...pipeline for,".
_DANGLING = (
    "and", "or", "but", "with", "for", "to", "of", "in", "on", "at", "by",
    "from", "as", "into", "using", "including", "such", "that", "which",
    "the", "a", "an", "our", "their", "its", "your", "this", "these", "over",
    "across", "through", "while", "when", "where", "plus", "via", "per",
)


def _trim_tail(text: str) -> str:
    """Drop trailing conjunctions and prepositions left by a length cut."""
    words = text.rstrip(" ,;:-").split()
    while words and words[-1].lower().strip(",;:-") in _DANGLING:
        words.pop()
    return " ".join(words).rstrip(" ,;:-")


def _shorten(text: str, limit: int) -> str:
    """Cut to a length without splitting a word or ending on a connective."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= limit:
        return _trim_tail(text)
    return _trim_tail(text[:limit].rsplit(" ", 1)[0])


def _label(name: str) -> str:
    """Category label for prose - lower-cased unless it is an acronym."""
    name = (name or "").strip()
    # Keep anything with an internal capital: acronyms (AI/ML) and camel-case
    # product names (DevOps) both read wrong in lower case.
    return name if any(part[1:] != part[1:].lower() for part in re.split(r"[/&\s]+", name) if part) \
        else name.lower()


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _tidy_duty(text: str, limit: int = 90) -> str:
    """A JD duty line, trimmed to something quotable inside a sentence."""
    text = re.sub(r"\s+", " ", (text or "").strip().rstrip(". "))
    # Postings address the reader directly ("At SmartSpot, you'll be
    # responsible for X"). Quoting that verbatim inside a first-person email
    # reads as if the sender were describing themselves in second person.
    text = re.sub(r"^at\s+[A-Z][\w&.\- ]{1,30},\s*", "", text, flags=re.I)
    text = re.sub(
        r"^(?:you\s+(?:will|would|can|should)\s+(?:be\s+)?|you'?ll\s+(?:be\s+)?|"
        r"the\s+candidate\s+will\s+|we\s+are\s+looking\s+for\s+|"
        r"in\s+this\s+role[, ]+|as\s+an?\s+[\w ]{2,30},\s*)",
        "", text, flags=re.I,
    )
    text = re.sub(r"^(?:be\s+)?responsible\s+for\s+", "", text, flags=re.I)
    text = _shorten(text, limit)
    if not text:
        return ""
    first = text.split(" ", 1)[0]
    # Leave acronyms and proper nouns alone; only de-capitalise ordinary words.
    if first.isupper() or (len(first) > 1 and first[1:].lower() != first[1:]):
        return text
    return text[:1].lower() + text[1:]


def _join_list(items, limit: int = 4, final: str = "and") -> str:
    items = [str(item).strip() for item in items if str(item).strip()][:limit]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" {final} " + items[-1]


def _role_sentence(context: dict[str, Any]) -> str:
    """Paragraph 1 - what the posting is asking for, in the JD's own terms."""
    role = context.get("role") or {}
    title = (context.get("job_title") or "").strip()
    company = context.get("company") or ""

    # Build the article into the phrase rather than around a placeholder -
    # falling back to the literal string "the role" produced "a the role" and
    # "the the role position".
    if context.get("mode") == "individual":
        opening = (
            f"I am writing to apply for the {title} position"
            if title else "I am writing to apply for this position"
        )
    else:
        opening = (
            f"We recently came across your requirement for {_article(title)} {title}"
            if title
            else "We recently came across your requirement for the advertised role"
        )
    opening += f" at {company}." if company else "."

    detail: list[str] = []
    skills = role.get("required_technical_skills") or []
    if skills:
        detail.append(f"centres on {_join_list(skills, 4)}")
    duties = [_tidy_duty(duty) for duty in (role.get("key_responsibilities") or [])]
    duties = [duty for duty in duties if len(duty.split()) >= 3]
    if duties:
        detail.append(f"covering work such as {duties[0]}")
    experience = role.get("experience_required") or ""
    if experience and experience.lower() not in ("not stated", ""):
        detail.append(f"and asks for {experience} of experience")

    if detail:
        opening += " As described, the role " + ", ".join(detail) + "."
    return opening


def _degree_phrase(value: str) -> str:
    """Turn the analysis label into something that reads in a sentence.

    Eligibility renders a degree as "Bachelor's - Data Science", which is
    right for a card and wrong inside prose.
    """
    value = (value or "").strip()
    if not value or value.lower() == "not stated":
        return ""
    if " - " in value:
        level, _, field = value.partition(" - ")
        return f"{level.strip()} in {field.strip()}"
    return value


def _individual_opening(context: dict[str, Any]) -> str:
    """Interest in the role, then straight to why this candidate fits.

    Deliberately not _role_sentence (which the team email still uses): that
    one explains the posting back to the reader, which is the wrong move in an
    application. A recruiter wrote the JD - they do not need it summarised.
    """
    title = (context.get("job_title") or "").strip()
    company = context.get("company") or ""
    education = (context.get("education") or {}).get("candidate_value") or ""
    skills = context.get("relevant_skills") or context.get("matched_skills") or []

    role_phrase = f"the {title} role" if title else "this role"
    opening = f"I am writing to express my interest in {role_phrase}"
    opening += f" at {company}." if company else "."

    credentials: list[str] = []
    degree = _degree_phrase(education)
    if degree:
        credentials.append(f"a {degree}")
    if skills:
        credentials.append(f"hands-on experience in {_join_list(skills, 3)}")

    if credentials:
        opening += (
            f" With {_join_list(credentials, 2)}, my background aligns closely with "
            "what the position calls for."
        )
    return opening


def _fallback_body_individual(context: dict[str, Any]) -> str:
    """An application, not a summary of the posting.

    Paragraph 1 states interest and the strongest credentials. Paragraph 2 is
    the evidence - real projects and real work from the CV. Paragraph 3 names
    only the skills this posting asks for. Every fact comes from the Phase One
    report; nothing absent from the CV is claimed, and nothing missing from it
    is listed.
    """
    matched = context.get("matched_skills") or []
    relevant = context.get("relevant_skills") or matched
    experience = context.get("experience") or {}
    role = context.get("role") or {}

    lines: list[str] = []

    # 1 - interest and fit
    lines.append(_individual_opening(context))
    lines.append("")

    # 2 - the evidence, in the candidate's own words
    para2: list[str] = []
    # De-duplicated: one requirement can match the same resume bullet more
    # than once, which previously printed the same project twice in a row.
    own: list[str] = []
    seen: set[str] = set()
    for item in context.get("candidate_evidence") or []:
        tidied = _tidy_duty(item, 130)
        key = tidied.lower()[:60]
        if len(tidied.split()) >= 3 and key not in seen:
            seen.add(key)
            own.append(tidied)
    if own:
        first = own[0]
        para2.append(f"Most relevantly, I {first}." if not first[:1].isupper()
                     else f"Most relevantly, {first}.")
        if len(own) > 1:
            para2.append(f"I have also {own[1]}.")
    candidate_years = experience.get("candidate_value") or ""
    if not own and candidate_years and candidate_years.lower() != "not stated":
        para2.append(f"I bring {candidate_years} of practical experience in this area.")
    if not para2:
        para2.append(
            "My CV sets out the projects and experience most relevant to this role."
        )

    # An experience gap is mentioned once, briefly, and only when the posting
    # asks for materially more than the candidate has. It is never the focus.
    required_years = role.get("experience_required") or ""
    if _significant_experience_gap(context) and own:
        para2.append(
            "My professional experience is at an earlier stage than the "
            f"{required_years} indicated, but the work above reflects direct, "
            "hands-on delivery in this area."
        )
    lines.append(" ".join(para2))
    lines.append("")

    # 3 - the skills this role asks for, not everything on the CV
    para3: list[str] = []
    if relevant:
        para3.append(f"Technically, I work most closely with {_join_list(relevant, 5)}.")
    elif matched:
        para3.append(f"Technically, I work with {_join_list(matched, 5)}.")
    lines.append(" ".join(para3))
    lines.append("")

    # 4 - the request, standing alone. Appended to the technical paragraph it
    # read as an afterthought; on its own it reads as the point of the email.
    lines.append(
        "It would be a pleasure to showcase my relevant work and discuss how my "
        "skills could contribute to your team, and I would be grateful if you "
        "could spare the time for a short meeting, whenever is most convenient "
        "for you. Thank you for your time and consideration."
    )

    return "\n".join(lines)


def _significant_experience_gap(context: dict[str, Any]) -> bool:
    """Whether the shortfall is large enough to be worth naming at all.

    A gap is only mentioned when the posting states a minimum, the CV states a
    total, and the shortfall is more than a year. Below that it is noise, and
    volunteering it weakens the application for no gain.
    """
    experience = context.get("experience") or {}
    if str(experience.get("status", "")).lower() not in ("fail", "below_requirement"):
        return False
    role = context.get("role") or {}
    return bool(role.get("experience_required")) and bool(experience.get("candidate_value"))


def _fallback_subject_team(context: dict[str, Any]) -> str:
    title = context.get("job_title") or ""
    company = context.get("company") or ""
    # Built from the capability the JD actually matched, so the subject line
    # changes with the posting instead of repeating one generic phrase.
    capabilities = context.get("logxperts_capabilities") or []
    capability = capabilities[0].get("name", "") if capabilities else ""
    if capability and title:
        return f"B2B Collaboration: {capability} for your {title} requirement"
    if capability:
        return f"Development Partnership: {capability}"
    if title:
        return f"B2B Collaboration: development support for your {title} requirement"
    if company:
        return f"Technology Partnership with {company}"
    return "B2B Development Partnership Enquiry"


def _fallback_body_team(context: dict[str, Any]) -> str:
    """Three paragraphs: the role, our relevant delivery record, our stack and
    how we would partner - closing on a B2B collaboration ask.

    Every capability and technology comes from the Logxperts
    profile. Nothing is asserted that the profile does not contain.
    """
    company = context.get("company") or ""
    org = context.get("organization_name") or "Logxperts"
    capabilities = context.get("logxperts_capabilities") or []
    projects = context.get("logxperts_projects") or []
    tech = context.get("logxperts_tech") or {}
    models = context.get("logxperts_engagement_models") or []
    differentiators = context.get("logxperts_differentiators") or []
    markets = context.get("logxperts_markets") or []

    lines: list[str] = []
    intro = _role_sentence(context)
    intro += f" I am writing from {org} regarding how we might support it."
    lines.append(intro)
    lines.append("")

    # 2 - our relevant experience and anonymized reference projects
    para2: list[str] = []
    if capabilities:
        cap_names = _join_list([cap.get("name", "") for cap in capabilities[:3]], 3)
        if cap_names:
            para2.append(f"This maps closely to our delivery capability in {cap_names}.")
        first = capabilities[0].get("description", "")
        if first:
            trimmed = _shorten(first, 180)
            para2.append(trimmed if trimmed.endswith(".") else trimmed + ".")
    if projects:
        described = projects[0]
        summary = described.get("summary", "")
        if summary:
            summary = _shorten(summary, 200)
            summary = summary if summary.endswith(".") else summary + "."
        if summary:
            para2.append(
                f"For example, our work on {described.get('name', 'a comparable project')}: "
                + (summary if summary.endswith(".") else summary + ".")
            )
    if not para2:
        para2.append(
            f"{org} delivers end-to-end software engineering, AI and data services "
            "for enterprise and government customers."
        )
    lines.append(" ".join(para2))
    lines.append("")

    # 3 - tech stack, how we engage, and the collaboration ask
    para3: list[str] = []
    if tech:
        grouped = "; ".join(
            f"{_label(category)}: {_join_list(items, 4)}"
            for category, items in list(tech.items())[:3]
        )
        para3.append(f"Our teams work across {grouped}.")
    if differentiators:
        para3.append(differentiators[0].get("description", ""))
    if markets:
        para3.append(f"We deliver across {_join_list(markets, 4)}.")
    if models:
        model_names = _join_list([model.get("name", "") for model in models[:3]], 3, final="or")
        if model_names:
            para3.append(
                f"We can engage through {model_names}, depending on how you would prefer to work."
            )
    lines.append(" ".join(part for part in para3 if part))
    lines.append("")

    # 4 - the collaboration request, on its own line for the same reason.
    lines.append(
        "It would be a pleasure to showcase some of our relevant work and discuss "
        f"how {org} could support your requirements"
        + (f" at {company}." if company else ".")
        + " We would be grateful if you could spare the time for a short meeting, "
        "whenever is most convenient for you."
    )

    return "\n".join(lines)


def _template_email(mode: str, context: dict[str, Any]) -> dict[str, str]:
    if mode == "team":
        return {
            "subject": _fallback_subject_team(context),
            "body": _fallback_body_team(context),
            "source": "template",
        }
    return {
        "subject": _fallback_subject_individual(context),
        "body": _fallback_body_individual(context),
        "source": "template",
    }


# ---------------------------------------------------------------------------
# Gemini-backed generation
# ---------------------------------------------------------------------------

_BANNED_PHRASES_NOTE = (
    "Avoid generic AI-sounding phrases such as \"thrilled\", \"excited to\", "
    "\"leveraging cutting-edge technologies\", \"synergy\", or \"best-in-class\" "
    "unless genuinely warranted. Write like a real, competent person, not a template."
)

_SYSTEM_PROMPT_INDIVIDUAL = f"""You write concise, professional job-application emails on behalf of an
individual candidate.

You will be given a JSON object describing the candidate and the role. That
JSON is the ONLY source of truth. Follow these rules strictly:

0. This is an application, not a summary of the posting. The recruiter WROTE
   the job description - do not explain it back to them. Never open with
   "This role involves...", "This position focuses on...", "The role
   requires..." or any similar restatement of the JD. Mention the role only
   to name what is being applied for, then move immediately to the
   candidate's own evidence. Roughly 80% of the email must be about the
   candidate.
1. Never state or imply any skill, technology, employer, certification,
   years of experience, degree, location, or willingness to relocate that is
   not explicitly present in the JSON.
2. Missing skills are omitted, not listed. Never claim a skill the JSON does
   not evidence, and never state that the candidate lacks something.
   An experience shortfall is mentioned ONLY when the posting states a
   minimum and the candidate is materially below it - and then in at most one
   short clause, positively framed, never as the subject of a sentence of its
   own. Do not write "While my professional timeline is shorter than..." or
   any similar construction that makes the gap the focus. If in doubt, leave
   it out entirely and let the evidence speak.
3. Do not claim a specific number of years of experience unless the JSON's
   "experience" field explicitly states it.
4. Do not claim the candidate is already located in the job's location, and
   do not claim willingness to relocate, unless the JSON says so.
5. Treat academic or personal projects as projects, never as professional
   employment, unless the JSON indicates otherwise.
6. If job_title or company is empty, phrase the email generically ("this
   position", "your team") rather than inventing one.
7. Structure the body as exactly FOUR short paragraphs, separated by blank
   lines. The fourth is the meeting request and must always stand alone -
   never merged into the paragraph above it.
   (1) One or two sentences: interest in the named role, then the
       candidate's strongest relevant credentials. Model the shape on -
       "I am writing to express my interest in the X role. With a
       <degree> and hands-on experience in <2-3 skills>, my background
       aligns closely with what the position calls for." Use the actual
       degree and skills from the JSON.
   (2) The strongest paragraph, and the heart of the email. Give concrete
       evidence from candidate_evidence, projects and experience: what was
       actually built, with which technologies, and to what effect. Explain
       a project as evidence of capability rather than naming a skill - a
       YOLOv8 instance-segmentation pipeline demonstrates practical computer
       vision pipeline development, so say that. Prefer specifics over
       adjectives. Never invent a detail to make a project sound larger.
   (3) Two or three sentences on the technical skills that matter for THIS
       role, taking them in order from relevant_skills (already ranked by
       what the posting asks for). Do not list every skill in the JSON, and
       do not write a comma-separated keyword string - make it read as prose.
       Close by offering to showcase the candidate's relevant work, to
       discuss how their skills could contribute to the team, and to ask for
       a meeting - for example "It would be a pleasure to showcase my
       relevant work and discuss how my skills could contribute to your team,
       and I would be grateful if you could spare the time for a short
       meeting, whenever is most convenient for you."
       The register matters: the candidate is ASKING the reader for their
       time, not granting them an opportunity. Avoid "I would appreciate the
       opportunity to..." and any wording that implies the reader benefits
       from the meeting. Prefer "it would be a pleasure", "I would be
       grateful if", "if you could spare the time". Never demanding - avoid
       "Please schedule a meeting" or "Let us know when you are available".
8. Never include ATS or analysis vocabulary. No scores, percentages,
   "keyword match", "missing skills", "partially matched", "ATS" or any
   similar term. The recruiter must receive an ordinary application email
   with no trace of the analysis behind it.
9. Do NOT write a greeting or a sign-off - the application adds
   "Dear Hiring Manager," and the candidate's own name itself. Start
   directly with the first paragraph.
10. Body length: approximately 150-220 words. Tone: professional, concise,
   confident, and human - not apologetic, not boastful.
   {_BANNED_PHRASES_NOTE}
11. The subject must name the actual role, e.g. "Application for Computer
   Vision Engineer Role". Never "Job Application", "Application for role",
   "Application for position", or a stray word taken from the JD.
12. Output ONLY a JSON object of the exact shape {{"subject": "...", "body":
   "..."}} - no markdown code fences, no commentary, no extra keys.
"""

_SYSTEM_PROMPT_TEAM = f"""You write concise, professional B2B outreach emails on behalf of Logxperts,
a technology services company, reaching out to a client
whose job/project posting describes a role they need to fill.

You will be given a JSON object with the job description context and a
JD-relevant subset of Logxperts' own capabilities, technology stack and
anonymized reference projects. That JSON is the ONLY source of truth. Follow these
rules strictly:

1. Never mention any Logxperts capability, technology, project, market or
   engagement model that is NOT present in the JSON. The usable fields are
   logxperts_capabilities, logxperts_tech, logxperts_projects,
   logxperts_engagement_models, logxperts_delivery_model,
   logxperts_differentiators and logxperts_markets. Do not invent or assume
   anything beyond them. Never name any external customer or project owner.
   In particular, do NOT claim vendor partnerships, certifications or
   accreditations - none are provided.
2. Do not copy the JD excerpt verbatim - paraphrase your understanding of
   the role in your own words.
3. Structure the body as exactly FOUR short paragraphs, separated by blank
   lines. The fourth is the meeting request and must always stand alone -
   never merged into the paragraph above it.
   (1) Open with "We recently came across your requirement for..." or
       "We recently came across your job posting for...", then name the
       actual role, say in one sentence what the client is looking for,
       mention the 2-3 most relevant technical requirements, and turn
       naturally toward why Logxperts may be able to help. Never open with
       "Logxperts offers/provides..." or "We understand you are
       seeking..." - both read as presumptuous. If no reliable job title is
       available write "the advertised role"; never use a stray JD adjective
       such as "flexible", "proactive" or "strong" as the title.
   (2) The most important paragraph. Follow this chain explicitly: the JD's
       requirement -> the Logxperts capability that matches it -> the
       anonymized reference project that demonstrates it. Use only the 1-3
       most relevant entries from logxperts_projects; never list them all.
       Avoid unsupported promotion
       such as "deep domain experience" - prefer evidence-based wording, for
       example "relevant experience through our Computer Vision & Intelligent
       Video capability".
   (3) Only the technologies from logxperts_tech that are relevant to THIS
       posting - not the whole stack. Then the engagement models from
       logxperts_engagement_models, and close by offering to showcase relevant
       Logxperts work, to discuss how Logxperts could support their
       requirements, and to arrange a meeting at the reader's convenience -
       for example "It would be a pleasure to showcase some of our relevant
       work and discuss how Logxperts could support your requirements,
       and we would be grateful if you could spare the time for a short
       meeting, whenever is most convenient for you."
       The register matters: Logxperts is ASKING the reader for their time, not
       granting them an opportunity. Avoid "we would appreciate the
       opportunity to..." and anything implying the reader benefits from the
       meeting. Prefer "it would be a pleasure", "we would be grateful if",
       "if you could spare the time". Never demanding - avoid "Please
       schedule a meeting" or "Let us know your availability".
       Never imply Logxperts is applying for employment.
4. If logxperts_projects is empty, do not name any specific client or project -
   describe capabilities generally instead.
5. The email must change meaningfully with the JD. A different posting must
   surface different capabilities, anonymized project descriptions and
   technologies - never one generic Logxperts paragraph reused for every role.
6. The subject must be specific to the opportunity, e.g. "B2B Collaboration:
   Computer Vision & Intelligent Video for your Computer Vision Engineer
   requirement". Never a generic subject.
7. Body length: approximately 200-300 words - three substantive paragraphs.
   Tone: professional, concise, consultative, not aggressive sales language.
   {_BANNED_PHRASES_NOTE}
6. Do NOT write a greeting or a sign-off - the application adds
   "Dear Hiring Manager," and "Best regards, Logxperts" itself.
   Start directly with the first paragraph.
7. This is a business development email. Never mention the candidate's ATS
   score, CV score, missing skills or individual eligibility - none of that
   belongs in a B2B approach.
7. Output ONLY a JSON object of the exact shape {{"subject": "...", "body":
   "..."}} - no markdown code fences, no commentary, no extra keys.
"""


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except (ValueError, TypeError):
            return None
    return None


def _call_gemini(mode: str, context: dict[str, Any], vary: bool = False) -> Optional[dict[str, str]]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    system_prompt = _SYSTEM_PROMPT_TEAM if mode == "team" else _SYSTEM_PROMPT_INDIVIDUAL
    user_prompt = "Context JSON:\n" + json.dumps(context, ensure_ascii=False)
    if vary:
        user_prompt += (
            "\n\nWrite a fresh draft with a different opening and structure than a "
            "typical first attempt, while still following every rule above."
        )

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.8 if vary else 0.6,
            # A 250-300 word email is ~450 tokens, but newer Gemini models
            # spend tokens on internal reasoning before emitting any visible
            # text. At 800 the reply was being cut off mid-sentence, leaving
            # unparseable JSON and a silent fall back to the template.
            "maxOutputTokens": 4096,
            # Ask for JSON directly rather than hoping for it. This removes
            # the code fences and stray prose the parser had to strip.
            "responseMimeType": "application/json",
        },
    }

    try:
        response = requests.post(
            GEMINI_URL,
            # Sent as a header as well as a query parameter. Newer Google API
            # keys are only accepted in the header, and passing both is
            # harmless for the older AIza-style keys.
            headers={"x-goog-api-key": api_key},
            params={"key": api_key},
            json=payload,
            timeout=GEMINI_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            # The response body carries Google's actual reason - an invalid
            # key, an unknown model, an exhausted quota. Returning None
            # without it made every failure look the same.
            logger.error(
                "Gemini HTTP %s for model %s: %s",
                response.status_code, GEMINI_MODEL, response.text[:400],
            )
            return None
        data = response.json()
        candidate = data["candidates"][0]
        finish = candidate.get("finishReason", "")
        if finish and finish not in ("STOP", "FINISH_REASON_STOP"):
            # MAX_TOKENS here means the reply was cut off, not malformed.
            logger.error("Gemini stopped early (finishReason=%s); raise maxOutputTokens.", finish)
        text = candidate["content"]["parts"][0]["text"]
    except requests.RequestException as exc:
        logger.error("Gemini request failed: %s", exc)
        return None
    except (KeyError, IndexError, ValueError) as exc:
        logger.error("Gemini response could not be read (%s): %s",
                     type(exc).__name__, str(data)[:400] if "data" in dir() else "")
        return None

    parsed = _extract_json_object(text)
    if not parsed or "subject" not in parsed or "body" not in parsed:
        logger.error("Gemini reply was not the expected JSON: %s", text[:300])
        return None

    subject = str(parsed["subject"]).strip()
    body = str(parsed["body"]).strip()
    if not subject or not body:
        return None

    return {"subject": subject, "body": body, "source": "gemini"}


GREETING = "Dear Hiring Manager,"


def _enforce_envelope(body: str, mode: str, context: dict[str, Any]) -> str:
    """Guarantee the greeting and sign-off on every email, whatever wrote it.

    The template always produced them; the model was inconsistent, sometimes
    opening straight into the company description. Rather than hoping a prompt
    line holds, the envelope is applied after generation - so both modes look
    the same no matter which path produced the text.
    """
    body = (body or "").strip()

    # Strip any greeting the model already wrote, so it is never duplicated.
    body = re.sub(r"^(?:dear\s+[^\n,]{0,40},?|hi\s+[^\n,]{0,30},?|hello[^\n,]{0,30},?)\s*\n*",
                  "", body, flags=re.I).lstrip()

    signer = (
        context.get("organization_name") or "Logxperts"
        if mode == "team"
        else (context.get("candidate_name") or "").strip()
    )

    # Strip a sign-off the model already wrote, so ours is the only one.
    body = re.sub(
        r"\n+\s*(?:best\s+regards|kind\s+regards|regards|sincerely|yours\s+\w+|thanks|thank\s+you)"
        r"\s*,?\s*\n*.{0,60}$",
        "", body, flags=re.I | re.S,
    ).rstrip()

    # A stray comma or semicolon before the sign-off reads as a typo.
    body = body.rstrip(" ,;")
    if body and body[-1] not in ".!?\"'":
        body += "."

    closing = f"Best regards,\n{signer}" if signer else "Best regards,"
    return f"{GREETING}\n\n{body}\n\n{closing}"


def generate_email_with_fallback(
    mode: str, context: dict[str, Any], force_llm_variation: bool = False
) -> dict[str, str]:
    """Always returns a usable {subject, body, source} dict, never raises."""
    mode = mode if mode in ("team", "individual") else "individual"
    result: Optional[dict[str, str]] = None
    try:
        result = _call_gemini(mode, context, vary=force_llm_variation)
        if not result:
            logger.info("Gemini returned no usable email; using the template instead.")
    except Exception:
        # Logged, not swallowed. A silent fallback made a missing key, an
        # exhausted quota and a network failure all look identical.
        logger.exception("Gemini call failed; using the template instead.")

    if not result:
        result = _template_email(mode, context)

    # Applied last, so the greeting and sign-off are identical whichever path
    # produced the text.
    result["body"] = _enforce_envelope(result.get("body", ""), mode, context)
    return result