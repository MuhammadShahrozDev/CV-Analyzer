from __future__ import annotations

import math
import re
from typing import Any

from .concepts import (
    is_meaningful_requirement,
    lexical_alignment,
    skill_evidence_bonus,
)
from .eligibility import screen_eligibility
from .extractor import extract_skills_from_text, normalize_skill_text
from .preprocessing import preprocess_text
from .semantic import all_best_matches, best_sentence_matches, cosine_similarity_score
from .sectioning import (
    extract_bullet_items,
    extract_sections,
    meaningful_text,
    section_text,
    sentence_chunks,
)

STOPWORD_LIKE = {
    "experience",
    "responsible",
    "worked",
    "work",
    "team",
    "project",
    "projects",
    "skill",
    "skills",
    "ability",
    "knowledge",
    "develop",
    "developed",
    "using",
    "used",
    "include",
    "including",
    "job",
    "description",
    "role",
    "candidate",
    # Additional job-posting filler that inflated the keyword denominator.
    "requirement",
    "requirements",
    "qualification",
    "qualifications",
    "responsibility",
    "responsibilities",
    "opportunity",
    "employer",
    "company",
    "benefit",
    "benefits",
    "salary",
    "apply",
    "application",
    "applicant",
    "position",
    "preferred",
    "required",
    "strong",
    "excellent",
    "good",
    "great",
    "proven",
    "demonstrated",
    "year",
    "years",
    "plus",
    "etc",
    "join",
    "looking",
    "seeking",
    # Posting metadata and perks. These describe the advert, not the work, and
    # counting them made a candidate look worse for not writing "hybrid" or
    # "insurance" on their resume.
    "location",
    "onsite",
    "remote",
    "hybrid",
    "office",
    "site",
    "insurance",
    "bonus",
    "leave",
    "hour",
    "hours",
    "shift",
    "contract",
    "permanent",
    "vacancy",
    "want",
    "need",
    "help",
    "new",
    "well",
    "environment",
}

TECHNICAL_CATEGORIES = {
    "programming languages",
    "databases",
    "visualization tools",
    "machine learning",
    "cloud platforms",
    "libraries",
    "engineering practices",
    "certifications",
}

SOFT_CATEGORIES = {"soft skills", "business skills"}

# Sections whose job-side counterpart is usually absent from a posting. When
# the job side is empty these fall back to the whole posting rather than
# scoring zero.
FALLBACK_SECTIONS = {
    "professional_summary",
    "technical_skills",
    "work_experience",
    "projects",
    "soft_skills",
}

# How much credit each kind of evidence earns when covering a requirement.
EVIDENCE_WEIGHTS = {
    "stated": 1.00,    # skill written explicitly in the resume
    "inferred": 0.80,  # implied by a tool the resume does state
    "partial": 0.50,   # near-miss wording
}

# A requirement is considered covered at or above this combined score.
REQUIREMENT_COVERED_AT = 0.50
REQUIREMENT_STRONG_AT = 0.70
REQUIREMENT_WEAK_AT = 0.30

# Every responsibility gets one of these, never a bare "no strong matches".
# A weak match with the bullet that came closest tells a candidate far more
# than silence does: it shows what the analyzer read as related, so they can
# see whether the gap is real or just wording.
MATCH_LEVELS = ("Strong Match", "Moderate Match", "Weak Match", "Missing")


def classify_requirement(score: float) -> str:
    """Turn a 0..1 requirement score into a recruiter-facing verdict."""
    if score >= REQUIREMENT_STRONG_AT:
        return "Strong Match"
    if score >= REQUIREMENT_COVERED_AT:
        return "Moderate Match"
    if score >= REQUIREMENT_WEAK_AT:
        return "Weak Match"
    return "Missing"

# How much a job-side skill counts toward the skill score, by where the posting
# put it. A "nice to have" is not a requirement, and treating it as one meant a
# candidate who met every must-have was still marked down for skipping the
# optional extras. Preferred skills are therefore reported in full but weigh
# almost nothing.
REQUIRED_TIER_WEIGHT = 1.00    # named under Requirements / Must have
REQUIRED_PARTIAL_WEIGHT = 0.60  # named there, but only as a near-miss phrase
PREFERRED_TIER_WEIGHT = 0.20   # named under Preferred / Nice to have / Bonus
CONTEXT_TIER_WEIGHT = 0.15     # mentioned somewhere else in the posting
INFERRED_TIER_WEIGHT = 0.45    # never named, only implied by the posting

TIER_REQUIRED = "required"
TIER_PREFERRED = "preferred"
TIER_CONTEXT = "context"
TIER_INFERRED = "inferred"

TIER_ORDER = (TIER_REQUIRED, TIER_PREFERRED, TIER_CONTEXT, TIER_INFERRED)

# How far a term belonging to a named skill outranks a merely frequent one.
SKILL_TERM_BOOST = 2.5

# A stated requirement is still a requirement when it is a soft skill, but it
# is not the same kind of requirement. "Strong communication" and "proficiency
# in SQL" were carrying identical weight, so a candidate could lose as much for
# not writing the word "adaptability" as for not knowing SQL. Soft requirements
# are reported in full, under their own heading, and weigh far less.
REQUIRED_SOFT_WEIGHT = 0.35

KIND_TECHNICAL = "technical"
KIND_SOFT = "soft"

# Where a hard requirement can legitimately come from.
#
# Requirement blocks state the bar outright ("Required Qualifications", "Basic
# Qualifications", "Must have"), and duty blocks imply it - you cannot be asked
# to write SQL queries without SQL being required.
_REQUIREMENT_JOB_SECTIONS = ("technical_skills", "education", "certifications")
_RESPONSIBILITY_JOB_SECTIONS = ("work_experience", "projects", "soft_skills")
_REQUIRED_JOB_SECTIONS = _REQUIREMENT_JOB_SECTIONS + _RESPONSIBILITY_JOB_SECTIONS

# professional_summary is deliberately absent from the tuple above. Everything
# a posting writes before its first heading lands there - the job title, and
# very often a paragraph of company marketing. A sentence such as "our platform
# runs on AWS and Kubernetes with a React frontend" describes the employer, not
# a hiring bar, and reading it as a requirement produced missing-skill lists
# full of things the posting never asked anyone to have.

# Education section scores keyed by the eligibility verdict. A degree that
# satisfies the posting is a pass, full stop - previously this section was
# scored by wording similarity against a job block that usually does not exist,
# so a perfectly qualified candidate scored zero on Education.
_EDUCATION_STATUS_SCORES = {
    "match": 92.0,
    "borderline": 62.0,
    "fail": 35.0,
    "unknown": 70.0,
}


# Short tokens that really are ATS keywords. Everything else under three
# characters is noise, so these are allowed back in by name.
_SHORT_KEYWORD_ALLOWLIST = {
    "ai", "ml", "bi", "ux", "ui", "qa", "db", "os", "hr", "3d",
    "r", "go", "c", "c#", "js", "ts", "sq",
}

# Tokens that survive lemmatisation but carry no matching signal: bare numbers
# and money ("50000", "80k", "3"), version and date fragments ("2024", "v2"),
# ordinals, and punctuation left behind by bullet or range characters.
# Words that survive lemmatisation and look like content, but tell a candidate
# nothing about what to put on a resume. They come from the connective tissue
# of a job advert - "or a related field", "or equivalent practical background",
# "we prefer candidates who look at the whole process" - and were crowding out
# the terms that actually matter.
#
# Blocking them affects the keyword lists only. Skill detection runs on the raw
# document through alias matching, so "Time Management" and "Process
# Improvement" are still recognised as skills even though "time" and "process"
# are useless as bare keywords.
_GENERIC_JOB_TERMS = {
    "background", "field", "fields", "look", "looks", "prefer", "prefers",
    "process", "processes", "time", "times", "related", "relate", "practical",
    "equivalent", "able", "ability", "abilities", "candidate", "candidates",
    "end", "ends", "area", "areas", "aspect", "aspects", "basis", "case",
    "cases", "detail", "details", "effort", "efforts", "focus", "general",
    "ideal", "ideally", "level", "levels", "manner", "matter", "number",
    "part", "parts", "range", "sense", "side", "sort", "stage", "stages",
    "term", "terms", "thing", "things", "type", "types", "value", "way",
    "ways", "willing", "willingness", "wide", "range", "variety", "overall",
    "additional", "appropriate", "effective", "efficient", "relevant",
    "successful", "suitable", "essential", "desirable", "familiar",
    "familiarity", "comfortable", "passionate", "motivated", "driven",
    "dynamic", "fast", "paced", "highly", "closely", "directly", "daily",
    "weekly", "monthly", "ongoing", "day", "days", "week", "weeks", "month",
    "months", "role", "roles", "team", "teams", "work", "working", "works",
}

# Ordinary English stopwords. spaCy already strips these during preprocessing,
# but only when its language data is loaded; if the model is missing and the
# pipeline falls back to a blank one, this keeps the keyword lists clean.
_COMMON_STOPWORDS = {
    "the", "and", "for", "with", "you", "your", "our", "their", "its", "this",
    "that", "these", "those", "will", "shall", "can", "may", "must", "should",
    "would", "could", "have", "has", "had", "been", "being", "are", "was",
    "were", "not", "but", "all", "any", "some", "more", "most", "such", "into",
    "from", "about", "over", "under", "than", "then", "also", "very", "each",
    "other", "others", "who", "whom", "which", "what", "when", "where", "while",
    "here", "there", "them", "they", "she", "his", "her", "him", "one", "two",
    "per", "via", "upon", "within", "across", "among", "between", "both",
}

_NUMERIC_LIKE = re.compile(r"^[0-9]+(?:st|nd|rd|th|k|m|bn)?$")
_HAS_DIGIT = re.compile(r"[0-9]")
_ALPHA_RUN = re.compile(r"[a-z]")


def is_meaningful_keyword(token: str) -> bool:
    """Whether a token is worth reporting to a candidate as an ATS keyword.

    Filters, in order: empty and punctuation-only fragments; numbers, salary
    figures and years; stopword-like filler; and anything too short to
    identify a skill unless it is a known technical abbreviation.

    Tokens mixing letters and digits are dropped too. Real skills that need a
    digit ("s3", "ec2", "html5") reach the report through skill matching,
    which understands them, rather than through raw keyword overlap, which
    would also surface "40s", "12x" and "2024".
    """
    token = (token or "").strip()
    if not token:
        return False
    if token in STOPWORD_LIKE or token in _COMMON_STOPWORDS:
        return False
    if token in _GENERIC_JOB_TERMS:
        return False
    if not _ALPHA_RUN.search(token):
        # Pure punctuation ("+", "-", "/") or a bare number.
        return False
    if _NUMERIC_LIKE.match(token) or _HAS_DIGIT.search(token):
        return False
    if len(token) <= 2:
        return token in _SHORT_KEYWORD_ALLOWLIST
    return True


def filter_keywords(tokens) -> list[str]:
    """Meaningful keywords only, de-duplicated, order preserved."""
    seen: set[str] = set()
    kept: list[str] = []
    for token in tokens:
        if token in seen or not is_meaningful_keyword(token):
            continue
        seen.add(token)
        kept.append(token)
    return kept


def _content_tokens(text: str) -> list[str]:
    """Keyword-bearing tokens, with counts preserved for frequency weighting."""
    return [token for token in preprocess_text(text).split() if is_meaningful_keyword(token)]


def _normalize_category_name(name: str) -> str:
    return normalize_skill_text(name).strip()


def _skill_category_lookup(skills_config: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for category, values in skills_config.get("categories", {}).items():
        for entry in values:
            skill_name = entry if isinstance(entry, str) else entry.get("name", "")
            if skill_name:
                lookup[str(skill_name).strip().lower()] = category
    return lookup


def _skill_weight_lookup(skills_config: dict[str, Any]) -> dict[str, float]:
    """Per-skill importance, declared in skills.json. Defaults to 1.0."""
    weights: dict[str, float] = {}
    for _category, values in skills_config.get("categories", {}).items():
        for entry in values:
            if isinstance(entry, str):
                continue
            name = str(entry.get("name", "")).strip()
            if name:
                weights[name.lower()] = float(entry.get("weight", 1.0))
    return weights


def _categorize_skills(skills: list[str], skills_config: dict[str, Any]) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {}
    lookup = _skill_category_lookup(skills_config)
    for skill in skills:
        category = lookup.get(skill.lower(), "Uncategorized")
        categories.setdefault(category, []).append(skill)
    return categories


def _section_similarity(resume_section: str, job_section: str) -> float:
    if not resume_section or not job_section:
        return 0.0
    return cosine_similarity_score(resume_section, job_section)


# Lines that state an eligibility bar rather than a duty. These are checked by
# the eligibility screen, so counting them again as unanswered "requirements"
# told a qualified candidate to address a degree they already hold.
_SCREENING_LINE = re.compile(
    r"^\W*(?:"
    r"(?:bachelor|master|phd|doctorate|bs|ms|b\.?sc|m\.?sc|degree|diploma)\b"
    r"|.*\b\d{1,2}\s*\+?\s*(?:\u2013|-|to)?\s*\d{0,2}\s*\+?\s*(?:years?|yrs?)\b.*experience"
    r"|(?:must\s+be\s+(?:based|located|willing))"
    r"|(?:location|work\s+mode|job\s+type|employment\s+type|salary|contract\s+type)\s*[:\-]"
    r")",
    re.I,
)


def _is_screening_line(text: str) -> bool:
    return bool(_SCREENING_LINE.match((text or "").strip()))


def _strip_screening_lines(text: str) -> str:
    """Remove eligibility lines from text used for keyword and similarity work.

    The office address, work mode, salary band, degree line and years-of-
    experience line are screened by utils/eligibility.py. Leaving them in the
    scoring text made them count twice, and made an otherwise identical posting
    score differently just because the office moved city.
    """
    return "\n".join(
        line
        for line in (text or "").replace("\r", "\n").split("\n")
        if line.strip() and not _is_screening_line(line)
    ).strip()


def _extract_job_responsibilities(job_bundle) -> list[str]:
    """Duty lines from the posting, excluding boilerplate and screening bars."""
    parts = [section_text(job_bundle, name) for name in _REQUIRED_JOB_SECTIONS]
    combined = "\n".join(part for part in parts if part).strip()

    # Only when the posting has no requirement or duty blocks at all does the
    # untitled preamble stand in for them. Otherwise a company blurb was being
    # listed as a duty the candidate had failed to answer.
    if not combined:
        combined = section_text(job_bundle, "professional_summary").strip()

    chunks = sentence_chunks(combined)
    return [
        chunk
        for chunk in chunks
        if is_meaningful_requirement(chunk) and not _is_screening_line(chunk)
    ]


def _extract_resume_achievements(resume_bundle) -> list[str]:
    """Things the candidate did.

    Deliberately excludes the technical skills block. A skills list proves a
    tool is claimed, not that a duty was performed -- feeding it in as evidence
    let a resume of bare nouns satisfy every responsibility and outscore real
    experience.
    """
    parts = [
        section_text(resume_bundle, "work_experience"),
        section_text(resume_bundle, "projects"),
        section_text(resume_bundle, "professional_summary"),
        section_text(resume_bundle, "certifications"),
        section_text(resume_bundle, "volunteer_experience"),
    ]
    combined = "\n".join(part for part in parts if part).strip()
    return [item for item in extract_bullet_items(combined) if item.strip()]


def _narrative_text(resume_bundle) -> str:
    """Experience, projects and summary - where claims become evidence."""
    return "\n".join(
        part
        for part in (
            section_text(resume_bundle, "work_experience"),
            section_text(resume_bundle, "projects"),
            section_text(resume_bundle, "professional_summary"),
        )
        if part
    ).strip()


def _substantiation(
    resume_bundle,
    required_skills: list[str],
    skills_config: dict[str, Any],
) -> tuple[float, list[str]]:
    """Fraction of required skills backed by narrative, plus the ones that are not.

    A skill named only in the skills block is a claim; a skill that also
    appears in experience or projects is evidence. Commercial tools penalise
    the former, and without this the analyzer is trivially gamed by stuffing.
    """
    if not required_skills:
        return 1.0, []

    narrative = _narrative_text(resume_bundle)
    if not narrative:
        return 0.0, sorted(required_skills)

    narrative_profile = extract_skills_from_text(narrative, skills_config)
    proven = set(narrative_profile.get("matched_skills", []) or []) | set(
        narrative_profile.get("inferred_skills", []) or []
    )

    unsubstantiated = sorted(skill for skill in required_skills if skill not in proven)
    hits = len(required_skills) - len(unsubstantiated)
    return hits / len(required_skills), unsubstantiated


# ---------------------------------------------------------------------------
# Skill overlap
# ---------------------------------------------------------------------------
def _evidence_index(resume_skills: dict[str, Any]) -> dict[str, tuple[str, float]]:
    """Map skill name -> (evidence kind, credit). Strongest evidence wins."""
    index: dict[str, tuple[str, float]] = {}
    for skill in resume_skills.get("partial_skills", []) or []:
        index[skill] = ("partial", EVIDENCE_WEIGHTS["partial"])
    for skill in resume_skills.get("inferred_skills", []) or []:
        index[skill] = ("inferred", EVIDENCE_WEIGHTS["inferred"])
    for skill in resume_skills.get("matched_skills", []) or []:
        index[skill] = ("stated", EVIDENCE_WEIGHTS["stated"])
    return index


def skill_kind(skill: str, skills_config: dict[str, Any]) -> str:
    """Whether a skill is a technical capability or a behavioural one.

    Read from the category the skill sits in inside skills.json, so the split
    stays configurable rather than hard-coded to a list of names.
    """
    category = _skill_category_lookup(skills_config).get(skill.lower(), "")
    return KIND_SOFT if category.strip().lower() in SOFT_CATEGORIES else KIND_TECHNICAL


def _named_skills(text: str, skills_config: dict[str, Any]) -> set[str]:
    """Skills a block of text names outright, ignoring implied concepts."""
    if not (text or "").strip():
        return set()
    profile = extract_skills_from_text(text, skills_config)
    return set(profile.get("matched_skills", []) or []) | set(
        profile.get("partial_skills", []) or []
    )


def build_job_requirements(
    job_skills: dict[str, Any],
    required_names: set[str],
    preferred_names: set[str],
    skills_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """What the posting asks for, tiered by *where in the posting it asked*.

    Only skills the job description names or implies ever appear here. The
    catalogue in skills.json is a vocabulary for reading documents, not a
    checklist to grade a resume against, so a skill the posting never mentions
    can never be reported as missing.

    Four tiers, and the tier is decided by provenance rather than by wording:

      required  - named in a requirement block, or in a duty the role performs
      preferred - named only under Preferred / Nice to have / Bonus
      context   - mentioned somewhere else entirely (company blurb, tech stack,
                  an aside). Real vocabulary, but never a hiring bar.
      inferred  - never written down, only implied by something that was

    Only the first tier carries full weight, and only the first tier can appear
    as a missing *required* skill.
    """
    weights = _skill_weight_lookup(skills_config)
    requirements: dict[str, dict[str, Any]] = {}

    kind_lookup = {
        skill: (
            KIND_SOFT
            if _skill_category_lookup(skills_config).get(skill.lower(), "").strip().lower()
            in SOFT_CATEGORIES
            else KIND_TECHNICAL
        )
        for skill in (
            set(job_skills.get("matched_skills", []) or [])
            | set(job_skills.get("partial_skills", []) or [])
            | set(job_skills.get("inferred_skills", []) or [])
        )
    }

    def register(skill: str, tier: str, multiplier: float) -> None:
        if not skill or skill in requirements:
            return
        base = weights.get(skill.lower(), 1.0)
        kind = kind_lookup.get(skill, KIND_TECHNICAL)
        # Only a technical requirement carries its tier weight in full. A soft
        # requirement is still listed and still counts, but it cannot move the
        # score the way a missing programming language does.
        if tier == TIER_REQUIRED and kind == KIND_SOFT:
            multiplier *= REQUIRED_SOFT_WEIGHT
        requirements[skill] = {
            "weight": round(base * multiplier, 4),
            "tier": tier,
            "kind": kind,
            "base_weight": base,
        }

    matched = set(job_skills.get("matched_skills", []) or [])
    partial = set(job_skills.get("partial_skills", []) or [])
    inferred = set(job_skills.get("inferred_skills", []) or [])

    # Required beats preferred beats context, so a skill named in more than one
    # place is graded by the strongest claim the posting makes for it.
    preferred_names = set(preferred_names) - set(required_names)

    def tier_of(skill: str) -> tuple[str, float, float]:
        if skill in required_names:
            return TIER_REQUIRED, REQUIRED_TIER_WEIGHT, REQUIRED_PARTIAL_WEIGHT
        if skill in preferred_names:
            return TIER_PREFERRED, PREFERRED_TIER_WEIGHT, PREFERRED_TIER_WEIGHT
        return TIER_CONTEXT, CONTEXT_TIER_WEIGHT, CONTEXT_TIER_WEIGHT

    for skill in sorted(matched):
        tier, full, _partial_weight = tier_of(skill)
        register(skill, tier, full)

    for skill in sorted(partial):
        tier, _full, partial_weight = tier_of(skill)
        register(skill, tier, partial_weight)

    # A concept the posting only implies inherits the standing of whatever
    # implied it. Snowflake sitting under "Nice to have" implies Data
    # Warehousing, and that implication cannot be more binding than the
    # optional skill it came from - otherwise skipping an extra would cost
    # more than the extra itself. The same holds for a tool named only in a
    # company blurb.
    evidence = job_skills.get("inferred_evidence", {}) or {}
    for skill in sorted(inferred):
        sources = set(evidence.get(skill, []) or [])
        if sources and sources <= preferred_names:
            register(skill, TIER_PREFERRED, PREFERRED_TIER_WEIGHT * INFERRED_TIER_WEIGHT)
        elif sources and not (sources & set(required_names)):
            register(skill, TIER_CONTEXT, CONTEXT_TIER_WEIGHT * INFERRED_TIER_WEIGHT)
        else:
            register(skill, TIER_INFERRED, INFERRED_TIER_WEIGHT)

    return requirements


def skills_by_tier(
    requirements: dict[str, dict[str, Any]],
    tier: str,
    kind: str | None = None,
) -> list[str]:
    """Skills in a tier, optionally narrowed to technical or soft."""
    return sorted(
        skill
        for skill, spec in requirements.items()
        if spec["tier"] == tier and (kind is None or spec.get("kind", KIND_TECHNICAL) == kind)
    )


def _score_skill_overlap(
    resume_skills: dict[str, Any],
    requirements: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Weighted coverage of the skills the posting actually asks for.

    Credit is proportional to how strong the resume's evidence is (stated beats
    implied beats near-miss wording) and to how firmly the posting asked
    (required beats preferred). Missing skills are returned split by tier so a
    missing must-have is never presented as equivalent to a skipped extra.
    """
    evidence = _evidence_index(resume_skills)

    matched: list[str] = []
    partial: list[str] = []
    missing: dict[str, list[str]] = {tier: [] for tier in TIER_ORDER}
    missing_required_soft: list[str] = []
    earned = 0.0
    total = 0.0

    for skill, spec in requirements.items():
        weight = float(spec["weight"])
        total += weight
        kind, credit = evidence.get(skill, ("none", 0.0))
        earned += weight * credit

        if kind == "stated":
            matched.append(skill)
        elif kind in {"inferred", "partial"}:
            partial.append(skill)
        else:
            # Kept apart on purpose. Only the required bucket is a real gap:
            # the posting asked for it and the resume does not evidence it.
            # The rest are an optional extra, a passing mention, and a concept
            # nobody wrote down.
            missing[spec["tier"]].append(skill)
            if spec["tier"] == TIER_REQUIRED and spec.get("kind") == KIND_SOFT:
                missing_required_soft.append(skill)

    if total <= 0:
        # Nothing parseable was asked for. Describe the resume itself rather
        # than reporting a misleading zero, capped so an unmatched posting is
        # never presented as a strong match.
        resume_matched = sorted(resume_skills.get("matched_skills", []) or [])
        resume_partial = sorted(resume_skills.get("partial_skills", []) or [])
        denominator = len(resume_matched) + len(resume_partial)
        score = 0.0 if denominator == 0 else round((len(resume_matched) / denominator) * 100, 2)
        return {
            "score": min(score, 55.0),
            "matched": resume_matched,
            "partial": resume_partial,
            "missing_required": [],
            "missing_required_technical": [],
            "missing_required_soft": [],
            "missing_preferred": [],
            "missing_context": [],
            "missing_inferred": [],
        }

    return {
        "score": round((earned / total) * 100, 2),
        "matched": sorted(matched),
        "partial": sorted(partial),
        "missing_required": sorted(missing[TIER_REQUIRED]),
        "missing_required_technical": sorted(
            set(missing[TIER_REQUIRED]) - set(missing_required_soft)
        ),
        "missing_required_soft": sorted(missing_required_soft),
        "missing_preferred": sorted(missing[TIER_PREFERRED]),
        "missing_context": sorted(missing[TIER_CONTEXT]),
        "missing_inferred": sorted(missing[TIER_INFERRED]),
    }


# ---------------------------------------------------------------------------
# Requirement level matching
# ---------------------------------------------------------------------------
def match_requirements(
    requirements: list[str],
    achievements: list[str],
    resume_effective_skills: set[str],
    skills_config: dict[str, Any],
) -> dict[str, Any]:
    """Score every job requirement against the strongest resume evidence.

    Evidence comes from three independent channels and the best one wins:
      lexical  - shared deliverable and verb intent ("develop dashboards" vs
                 "built Power BI dashboards")
      semantic - embedding similarity, which catches paraphrase the lexical
                 layer misses
      skills   - the resume proves a tool that implies the requested concept

    Combining by maximum matters. Each channel has blind spots, and requiring
    agreement would reintroduce the strictness this is meant to fix.
    """
    if not requirements:
        return {
            "score": 0.0,
            "coverage": 0.0,
            "mean_quality": 0.0,
            "details": [],
            "covered": [],
            "uncovered": [],
            "evaluated": 0,
            "by_level": {level: [] for level in MATCH_LEVELS},
            "level_counts": {level: 0 for level in MATCH_LEVELS},
        }

    semantic_best: dict[str, dict[str, Any]] = {}
    if achievements:
        for entry in all_best_matches(requirements, achievements):
            semantic_best[str(entry["source"])] = entry

    details: list[dict[str, Any]] = []
    covered: list[str] = []
    uncovered: list[str] = []
    total = 0.0

    for requirement in requirements:
        lexical_score = 0.0
        best_evidence = ""
        for achievement in achievements:
            score = lexical_alignment(requirement, achievement)
            if score > lexical_score:
                lexical_score = score
                best_evidence = achievement

        semantic_entry = semantic_best.get(requirement)
        semantic_score = float(semantic_entry["raw"]) if semantic_entry else 0.0
        semantic_score = max(0.0, min(1.0, semantic_score))

        # Inference is genuine evidence but weaker than a written statement,
        # so it cannot on its own certify a requirement as fully met.
        inferred_score = 0.85 * skill_evidence_bonus(requirement, resume_effective_skills, skills_config)

        combined = max(lexical_score, semantic_score, inferred_score)
        total += combined

        if lexical_score >= semantic_score and best_evidence:
            evidence_text = best_evidence
        elif semantic_entry:
            evidence_text = str(semantic_entry["target"])
        else:
            evidence_text = best_evidence

        level = classify_requirement(combined)

        # Say where the evidence came from, so a candidate can tell a real
        # match from a lucky paraphrase.
        if level == "Missing":
            evidence_text = ""
            evidence_source = "No supporting evidence found in the resume."
        elif inferred_score >= max(lexical_score, semantic_score):
            evidence_source = "Implied by skills stated elsewhere in the resume."
        elif lexical_score >= semantic_score:
            evidence_source = "Matched to a resume bullet."
        else:
            evidence_source = "Matched by meaning rather than wording."

        detail = {
            "requirement": requirement,
            "score": round(combined * 100, 2),
            "lexical": round(lexical_score * 100, 2),
            "semantic": round(semantic_score * 100, 2),
            "inferred": round(inferred_score * 100, 2),
            "evidence": evidence_text,
            "evidence_source": evidence_source,
            "level": level,
            "covered": combined >= REQUIREMENT_COVERED_AT,
            "strong": combined >= REQUIREMENT_STRONG_AT,
        }
        details.append(detail)
        (covered if detail["covered"] else uncovered).append(requirement)

    evaluated = len(requirements)
    mean_score = total / evaluated
    coverage = len(covered) / evaluated

    # Mean quality and breadth both matter: a resume answering half the duties
    # brilliantly is not a better fit than one answering all of them well.
    blended = (0.6 * mean_score) + (0.4 * coverage)

    ordered = sorted(details, key=lambda item: item["score"], reverse=True)
    by_level: dict[str, list[dict[str, Any]]] = {level: [] for level in MATCH_LEVELS}
    for detail in ordered:
        by_level[detail["level"]].append(detail)

    return {
        "score": round(blended * 100, 2),
        "coverage": round(coverage * 100, 2),
        "mean_quality": round(mean_score * 100, 2),
        "details": ordered,
        "covered": covered,
        "uncovered": uncovered,
        "evaluated": evaluated,
        # Every responsibility appears in exactly one of these four groups.
        "by_level": by_level,
        "level_counts": {level: len(items) for level, items in by_level.items()},
    }


# ---------------------------------------------------------------------------
# Section analysis
# ---------------------------------------------------------------------------
def _score_by_category(skills: list[str], skills_config: dict[str, Any], wanted_categories: set[str]) -> list[str]:
    category_lookup = _skill_category_lookup(skills_config)
    matched = []
    for skill in skills:
        category = category_lookup.get(skill.lower(), "").strip().lower()
        if category in wanted_categories:
            matched.append(skill)
    return matched


def _education_section_score(
    education_assessment: dict[str, Any],
    semantic_score: float,
    section_present: bool,
) -> float:
    """Score Education on whether the degree qualifies, not on wording.

    Postings almost never contain an "Education" heading, so the old
    similarity-against-an-empty-string approach scored a fully qualified
    candidate at zero. The verdict from the eligibility screen is reused here,
    which is why a BS in Data Science now satisfies "BS Computer Science or
    related field" instead of being marked down for the different wording.
    """
    status = (education_assessment or {}).get("status", "unknown")
    if not section_present:
        return 40.0 if status == "unknown" else 30.0
    base = _EDUCATION_STATUS_SCORES.get(status, 70.0)
    return round(max(0.0, min(100.0, base + 0.08 * semantic_score)), 2)


def _analyze_section(
    section_name: str,
    resume_text: str,
    job_text: str,
    resume_skills: dict[str, Any],
    requirements: dict[str, dict[str, Any]],
    skills_config: dict[str, Any],
    responsibilities: list[str],
    achievements: list[str],
    *,
    job_fallback_text: str = "",
    section_resume_skills: dict[str, Any] | None = None,
    requirement_report: dict[str, Any] | None = None,
    education_assessment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Postings rarely carry resume-style headings. Comparing against an empty
    # string forced these sections to zero, and they hold most of the weight,
    # which is why well-matched resumes scored under 40%.
    effective_job_text = job_text
    used_fallback = False
    if not effective_job_text.strip() and section_name in FALLBACK_SECTIONS and job_fallback_text.strip():
        effective_job_text = job_fallback_text
        used_fallback = True

    semantic_score = _section_similarity(resume_text, effective_job_text)

    # Skill evidence local to this section, so Technical Skills is judged on
    # what the skills block contains rather than on one document-wide number.
    local_skills = section_resume_skills if section_resume_skills is not None else resume_skills
    local_overlap = _score_skill_overlap(local_skills, requirements)
    skill_score = local_overlap["score"]
    matched_skills = local_overlap["matched"]
    partial_skills = local_overlap["partial"]

    # Whole-document skill picture, for sections that legitimately draw on the
    # entire resume.
    global_overlap = _score_skill_overlap(resume_skills, requirements)
    global_skill_score = global_overlap["score"]
    global_missing = global_overlap["missing_required"]

    all_resume_skills = (resume_skills.get("matched_skills", []) or []) + (resume_skills.get("partial_skills", []) or [])
    all_job_skills = skills_by_tier(requirements, TIER_REQUIRED) + skills_by_tier(requirements, TIER_PREFERRED)
    technical_skills = _score_by_category(all_resume_skills, skills_config, TECHNICAL_CATEGORIES)
    job_technical_skills = _score_by_category(all_job_skills, skills_config, TECHNICAL_CATEGORIES)
    soft_skills = _score_by_category(all_resume_skills, skills_config, SOFT_CATEGORIES)
    job_soft_skills = _score_by_category(all_job_skills, skills_config, SOFT_CATEGORIES)

    responsibility_score = 0.0
    responsibility_matches: list[dict[str, Any]] = []
    if section_name in {"work_experience", "projects", "professional_summary"}:
        if requirement_report and requirement_report.get("evaluated"):
            responsibility_score = float(requirement_report.get("score", 0.0))
            responsibility_matches = [
                {
                    "source": item["requirement"],
                    "target": item["evidence"],
                    "score": item["score"],
                }
                for item in requirement_report.get("details", [])
                if item.get("covered")
            ]
        else:
            responsibility_matches = best_sentence_matches(responsibilities, achievements, threshold=0.55)
            if responsibility_matches:
                responsibility_score = round(
                    sum(float(match["score"]) for match in responsibility_matches) / len(responsibility_matches), 2
                )

    if section_name == "technical_skills":
        # A skills block is judged mostly on whether it names what was asked
        # for. Wording similarity is a minor signal. Tools listed elsewhere in
        # the resume still count, at a small discount.
        coverage = max(skill_score, 0.85 * global_skill_score)
        score = round((0.75 * coverage) + (0.25 * semantic_score), 2)
    elif section_name == "work_experience":
        score = round((0.55 * responsibility_score) + (0.20 * semantic_score) + (0.25 * global_skill_score), 2)
    elif section_name == "projects":
        score = round((0.45 * responsibility_score) + (0.25 * semantic_score) + (0.30 * global_skill_score), 2)
    elif section_name == "professional_summary":
        score = round((0.45 * semantic_score) + (0.35 * global_skill_score) + (0.20 * responsibility_score), 2)
    elif section_name == "education":
        score = _education_section_score(
            education_assessment or {}, semantic_score, bool(resume_text.strip())
        )
    elif section_name == "certifications":
        score = round((0.5 * semantic_score) + (0.5 * skill_score), 2)
    elif section_name == "volunteer_experience":
        score = round((0.5 * semantic_score) + (0.5 * skill_score), 2)
    else:
        score = round((0.5 * semantic_score) + (0.5 * global_skill_score), 2)

    return {
        "score": max(0.0, min(100.0, score)),
        "semantic_score": semantic_score,
        "skill_score": skill_score,
        "global_skill_score": global_skill_score,
        "responsibility_score": responsibility_score,
        "responsibility_matches": responsibility_matches[:8],
        "matched_skills": matched_skills,
        "partial_skills": partial_skills,
        "missing_skills": global_missing,
        "missing_required_technical_skills": global_overlap["missing_required_technical"],
        "missing_required_soft_skills": global_overlap["missing_required_soft"],
        "missing_preferred_skills": global_overlap["missing_preferred"],
        "missing_context_skills": global_overlap["missing_context"],
        "missing_inferred_skills": global_overlap["missing_inferred"],
        "technical_skills": technical_skills,
        "job_technical_skills": job_technical_skills,
        "soft_skills": soft_skills,
        "job_soft_skills": job_soft_skills,
        "resume_text": resume_text,
        "job_text": job_text,
        "resume_word_count": len(resume_text.split()),
        "job_word_count": len(job_text.split()),
        "used_job_fallback": used_fallback,
        "section_present": bool(resume_text.strip()),
    }


# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------
def calculate_keyword_match(resume_text: str, job_text: str, job_skill_terms: set[str] | None = None) -> dict[str, Any]:
    """Importance-weighted keyword coverage.

    Previously every job token counted equally, so a posting's filler diluted
    the score as much as its real requirements mattered. Tokens now carry
    weight by repetition and by whether they belong to a recognised skill.
    """
    resume_tokens = set(_content_tokens(resume_text))
    job_tokens = _content_tokens(job_text)
    if not job_tokens:
        return {"score": 0.0, "matched": [], "missing": []}

    frequency: dict[str, int] = {}
    for token in job_tokens:
        frequency[token] = frequency.get(token, 0) + 1

    skill_terms = job_skill_terms or set()

    # A term belonging to a recognised skill outranks ordinary repetition, so
    # the lists a candidate reads lead with technical vocabulary rather than
    # whichever ordinary word the advert happened to repeat most.
    weights: dict[str, float] = {}
    for token, count in frequency.items():
        weight = 1.0 + min(1.0, 0.7 * math.log1p(count - 1))
        if token in skill_terms:
            weight *= SKILL_TERM_BOOST
        weights[token] = weight

    matched = [token for token in frequency if token in resume_tokens]
    missing = [token for token in frequency if token not in resume_tokens]

    total_weight = sum(weights.values())
    earned_weight = sum(weights[token] for token in matched)
    score = round((earned_weight / total_weight) * 100, 2) if total_weight else 0.0

    # Surface the terms that cost the most first.
    matched.sort(key=lambda token: weights[token], reverse=True)
    missing.sort(key=lambda token: weights[token], reverse=True)

    return {"score": score, "matched": matched, "missing": missing}


def calculate_skill_match(
    resume_skills: dict[str, Any],
    requirements: dict[str, dict[str, Any]],
    skills_config: dict[str, Any],
) -> dict[str, Any]:
    overlap = _score_skill_overlap(resume_skills, requirements)
    categories = _categorize_skills(overlap["matched"] + overlap["partial"], skills_config)
    return {
        "score": overlap["score"],
        "matched": overlap["matched"],
        "partial": overlap["partial"],
        # ``missing`` stays required-only, so every existing consumer keeps its
        # original meaning: a gap the posting actually insists on.
        "missing": overlap["missing_required"],
        "missing_required": overlap["missing_required"],
        "missing_required_technical": overlap["missing_required_technical"],
        "missing_required_soft": overlap["missing_required_soft"],
        "missing_preferred": overlap["missing_preferred"],
        "missing_context": overlap["missing_context"],
        "missing_inferred": overlap["missing_inferred"],
        "required_skills": skills_by_tier(requirements, TIER_REQUIRED),
        "required_technical_skills": skills_by_tier(requirements, TIER_REQUIRED, KIND_TECHNICAL),
        "required_soft_skills": skills_by_tier(requirements, TIER_REQUIRED, KIND_SOFT),
        "preferred_skills": skills_by_tier(requirements, TIER_PREFERRED),
        "context_skills": skills_by_tier(requirements, TIER_CONTEXT),
        "inferred_requirements": skills_by_tier(requirements, TIER_INFERRED),
        "matched_categories": categories,
    }


def calculate_cosine_similarity_score(resume_text: str, job_text: str) -> float:
    return cosine_similarity_score(resume_text, job_text)


def build_match_report(
    *,
    resume_text: str,
    job_text: str,
    resume_clean: str,  # kept for signature compatibility; see note below
    job_clean: str,     # kept for signature compatibility; see note below
    resume_skills: dict[str, Any],
    job_skills: dict[str, Any],
    skills_config: dict[str, Any],
    resume_layout_text: str,
    job_layout_text: str,
) -> dict[str, Any]:
    # resume_clean / job_clean remain in the signature so app.py and the test
    # suites keep working unchanged. They are no longer used to build the
    # keyword lists: those now come from calculate_keyword_match below, which
    # already applies section filtering and importance weighting.
    resume_bundle = extract_sections(resume_layout_text)
    job_bundle = extract_sections(job_layout_text, document_type="job")

    # Benefits, culture blurbs and legal boilerplate are excluded from every
    # comparison; they add unrelated vocabulary and depress similarity.
    # The preferred block is excluded here as well as in the skill tiers, so a
    # nice-to-have tool cannot come back as a "missing keyword" either.
    job_meaningful = (
        _strip_screening_lines(meaningful_text(job_bundle, exclude={"preferred_qualifications"}))
        or job_layout_text
    )
    resume_meaningful = meaningful_text(resume_bundle) or resume_layout_text

    # Tier the posting's own skills by which block of the posting names them.
    # This is the whole of the required/preferred/context separation: a skill
    # is a requirement because of where it was written, not because it happens
    # to be in the catalogue.
    preferred_text = section_text(job_bundle, "preferred_qualifications")
    required_text = "\n".join(
        part
        for part in (section_text(job_bundle, name) for name in _REQUIRED_JOB_SECTIONS)
        if part
    ).strip()

    # A posting with no headings at all has no requirement block to read, and
    # scoring it against nothing would be worse than reading it loosely. Only
    # in that case does the whole posting stand in as the requirement source.
    if not required_text:
        required_text = job_meaningful

    required_names = _named_skills(required_text, skills_config)
    preferred_names = _named_skills(preferred_text, skills_config) - required_names

    job_requirements = build_job_requirements(
        job_skills, required_names, preferred_names, skills_config
    )

    job_skill_terms: set[str] = set()
    for skill in (job_skills.get("matched_skills", []) or []):
        job_skill_terms.update(preprocess_text(skill).split())

    # Keyword advice is drawn from the requirement and duty blocks, the same
    # place required skills come from. Scoring it against the whole posting
    # told candidates to echo the employer's own marketing - "use more of the
    # posting's wording, such as acme, fintech, kubernetes" - none of which
    # belongs on their resume.
    job_keyword_text = _strip_screening_lines(required_text) or job_meaningful
    keyword_match = calculate_keyword_match(resume_text, job_keyword_text, job_skill_terms)
    skill_match = calculate_skill_match(resume_skills, job_requirements, skills_config)
    similarity = calculate_cosine_similarity_score(resume_meaningful, job_meaningful)

    # The displayed keyword lists are the scored ones. They used to be built
    # from a second, unfiltered pass over the raw lemma sets, which is why
    # salary figures, years and punctuation fragments appeared as "missing
    # keywords" the candidate was implicitly told to add - and why the list on
    # screen could disagree with the keyword score beside it. Both now come
    # from calculate_keyword_match, already filtered and ordered by how much
    # each term actually costs.
    matched_keywords = keyword_match["matched"]
    missing_keywords = keyword_match["missing"]

    job_responsibilities = _extract_job_responsibilities(job_bundle)
    resume_achievements = _extract_resume_achievements(resume_bundle)

    resume_effective = set(resume_skills.get("effective_skills", []) or resume_skills.get("matched_skills", []) or [])
    requirement_report = match_requirements(
        job_responsibilities, resume_achievements, resume_effective, skills_config
    )

    required_skill_names = sorted(
        set(skills_by_tier(job_requirements, TIER_REQUIRED))
        & set(resume_skills.get("matched_skills", []) or [])
    )
    substantiation, unsubstantiated_skills = _substantiation(
        resume_bundle, required_skill_names, skills_config
    )

    # Hard filters a recruiter applies before reading the score at all. This
    # never touches the ATS number - see utils/eligibility.py.
    eligibility = screen_eligibility(
        resume_text=resume_layout_text,
        job_text=job_layout_text,
        resume_experience_text="\n".join(
            part
            for part in (
                section_text(resume_bundle, "work_experience"),
                section_text(resume_bundle, "projects"),
            )
            if part
        ),
        resume_education_text=section_text(resume_bundle, "education"),
        job_education_text=section_text(job_bundle, "education"),
    )
    education_assessment = eligibility.get("education_check", {})

    section_texts = {
        "professional_summary": section_text(resume_bundle, "professional_summary"),
        "technical_skills": section_text(resume_bundle, "technical_skills"),
        "work_experience": section_text(resume_bundle, "work_experience"),
        "projects": section_text(resume_bundle, "projects"),
        "education": section_text(resume_bundle, "education"),
        "certifications": section_text(resume_bundle, "certifications"),
        "volunteer_experience": section_text(resume_bundle, "volunteer_experience"),
        "soft_skills": section_text(resume_bundle, "soft_skills"),
    }

    # Screening lines are removed from the job side of every section too, so a
    # posting that only differs by its office address cannot produce a
    # different section score.
    job_section_texts = {
        "professional_summary": _strip_screening_lines(section_text(job_bundle, "professional_summary")),
        "technical_skills": _strip_screening_lines(section_text(job_bundle, "technical_skills")),
        "work_experience": _strip_screening_lines(section_text(job_bundle, "work_experience")),
        "projects": _strip_screening_lines(section_text(job_bundle, "projects")),
        "education": _strip_screening_lines(section_text(job_bundle, "education")),
        "certifications": _strip_screening_lines(section_text(job_bundle, "certifications")),
        "volunteer_experience": _strip_screening_lines(section_text(job_bundle, "volunteer_experience")),
        "soft_skills": _strip_screening_lines(section_text(job_bundle, "soft_skills")),
    }

    # Skills scoped to each resume section, so section scores reflect that
    # section's own evidence instead of repeating one document-wide number.
    section_skill_profiles = {
        name: (extract_skills_from_text(text, skills_config) if text.strip() else None)
        for name, text in section_texts.items()
    }

    section_analysis = {
        section_name: _analyze_section(
            section_name,
            section_texts.get(section_name, ""),
            job_section_texts.get(section_name, ""),
            resume_skills,
            job_requirements,
            skills_config,
            job_responsibilities,
            resume_achievements,
            job_fallback_text=job_meaningful,
            section_resume_skills=section_skill_profiles.get(section_name),
            requirement_report=requirement_report,
            education_assessment=education_assessment,
        )
        for section_name in section_texts
    }

    def _as_match(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "source": item["requirement"],
            "target": item.get("evidence", ""),
            "score": item["score"],
            "level": item.get("level", ""),
            "evidence_source": item.get("evidence_source", ""),
        }

    details = requirement_report.get("details", [])
    strong_matches = [_as_match(item) for item in details if item.get("strong")]
    partial_matches = [
        _as_match(item) for item in details if item.get("covered") and not item.get("strong")
    ]
    # Every responsibility, grouped by verdict, so the report can show a Weak
    # Match with its closest bullet instead of an unhelpful "no strong matches".
    responsibility_breakdown = {
        level: [_as_match(item) for item in items]
        for level, items in (requirement_report.get("by_level", {}) or {}).items()
    }

    return {
        "keyword_match": keyword_match,
        "skill_match": skill_match,
        "similarity": similarity,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "resume_skills": resume_skills.get("matched_skills", []),
        "job_skills": job_skills.get("matched_skills", []),
        "resume_partial_skills": resume_skills.get("partial_skills", []),
        "job_partial_skills": job_skills.get("partial_skills", []),
        "resume_missing_skills": resume_skills.get("missing_skills", []),
        "section_analysis": section_analysis,
        "responsibility_matches": strong_matches,
        "partial_responsibility_matches": partial_matches,
        "responsibility_breakdown": responsibility_breakdown,
        "responsibility_level_counts": requirement_report.get("level_counts", {}),
        "resume_sections": section_texts,
        "job_sections": job_section_texts,
        "skills_config": skills_config,
        "resume_layout_text": resume_layout_text,
        "job_layout_text": job_layout_text,
        # Additive detail used by the scorer for explanatory feedback.
        "requirement_report": requirement_report,
        "resume_inferred_skills": resume_skills.get("inferred_skills", []),
        "resume_inferred_evidence": resume_skills.get("inferred_evidence", {}),
        "job_inferred_skills": job_skills.get("inferred_skills", []),
        "resume_effective_skills": sorted(resume_effective),
        "substantiation": round(substantiation, 4),
        # Computed once alongside the ratio above rather than re-extracting the
        # whole narrative a third time.
        "unsubstantiated_skills": unsubstantiated_skills,
        # Requirement tiers, so the report can show what is genuinely required
        # separately from what is merely preferred.
        "job_requirements": job_requirements,
        "required_skills": skills_by_tier(job_requirements, TIER_REQUIRED),
        # Task 4: the two kinds of requirement, reported separately so a
        # behavioural expectation is never presented as a hard technical gap.
        "required_technical_skills": skills_by_tier(job_requirements, TIER_REQUIRED, KIND_TECHNICAL),
        "required_soft_skills": skills_by_tier(job_requirements, TIER_REQUIRED, KIND_SOFT),
        "preferred_skills": skills_by_tier(job_requirements, TIER_PREFERRED),
        "inferred_requirements": skills_by_tier(job_requirements, TIER_INFERRED),
        "context_skills": skills_by_tier(job_requirements, TIER_CONTEXT),
        "missing_required_skills": skill_match.get("missing_required", []),
        "missing_required_technical_skills": skill_match.get("missing_required_technical", []),
        "missing_required_soft_skills": skill_match.get("missing_required_soft", []),
        "missing_preferred_skills": skill_match.get("missing_preferred", []),
        "missing_context_skills": skill_match.get("missing_context", []),
        "missing_inferred_skills": skill_match.get("missing_inferred", []),
        # Eligibility screening, reported separately and never folded into the
        # ATS score.
        "eligibility": eligibility,
        "education_assessment": education_assessment,
        "job_has_sections": any(
            job_section_texts.get(name, "").strip() for name in ("technical_skills", "work_experience")
        ),
    }