from __future__ import annotations

import re
from dataclasses import dataclass

SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "professional_summary": (
        "professional summary",
        "summary",
        "profile",
        "professional profile",
        "career summary",
        "objective",
    ),
    "technical_skills": (
        "technical skills",
        "skills",
        "core skills",
        "technical competencies",
        "core competencies",
        "tools and technologies",
        "technical expertise",
    ),
    "work_experience": (
        "work experience",
        "professional experience",
        "experience",
        "employment history",
        "work history",
        "professional background",
        "career history",
    ),
    "projects": (
        "projects",
        "project experience",
        "selected projects",
        "academic projects",
        "personal projects",
    ),
    "education": (
        "education",
        "academic background",
        "academic qualifications",
        "qualifications",
        "degree",
    ),
    "certifications": (
        "certifications",
        "certificates",
        "licenses",
        "training",
    ),
    "volunteer_experience": (
        "volunteer experience",
        "volunteering",
        "community service",
        "community involvement",
    ),
    "soft_skills": (
        "soft skills",
        "core competencies",
        "key strengths",
        "interpersonal skills",
    ),
}

SECTION_DISPLAY_NAMES: dict[str, str] = {
    "professional_summary": "Professional Summary",
    "technical_skills": "Technical Skills",
    "work_experience": "Work Experience",
    "projects": "Projects",
    "education": "Education",
    "certifications": "Certifications",
    "volunteer_experience": "Volunteer Experience",
    "soft_skills": "Soft Skills",
}

SECTION_ORDER = list(SECTION_DISPLAY_NAMES.keys())

# Job descriptions never use resume headings. Without this vocabulary a JD
# headed "Responsibilities:" / "Requirements:" collapsed entirely into
# professional_summary, leaving technical_skills, work_experience and projects
# empty on the job side -- and those three carry 80% of the ATS weight, so
# every comparison against them scored zero. JD headings are mapped onto the
# same canonical keys so the two documents can be compared section to section.
JOB_SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "work_experience": (
        "responsibilities",
        "key responsibilities",
        "main responsibilities",
        "core responsibilities",
        "duties",
        "key duties",
        "the role",
        "about the role",
        "role description",
        "job description",
        "what you will do",
        "what you'll do",
        "what you will be doing",
        "your role",
        "day to day",
        "day-to-day",
        "job duties",
        "position summary",
        "scope of work",
    ),
    "technical_skills": (
        "requirements",
        "required skills",
        "required qualifications",
        "qualifications",
        "minimum qualifications",
        "basic qualifications",
        "skills and experience",
        "skills required",
        "technical requirements",
        "technical skills",
        "must have",
        "must haves",
        "what we are looking for",
        "what we're looking for",
        "who you are",
        "your profile",
        "candidate profile",
        "experience required",
    ),
    # Optional extras are kept apart from hard requirements. Folding them into
    # the same bucket meant a missing "nice to have" cost the candidate as much
    # as a missing must-have, which is not how a posting is meant to be read.
    "preferred_qualifications": (
        "preferred qualifications",
        "preferred skills",
        "preferred experience",
        "nice to have",
        "nice to haves",
        "nice-to-have",
        "bonus points",
        "bonus skills",
        "desirable",
        "desirable skills",
        "good to have",
        "advantageous",
        "a plus",
        "pluses",
        "we would love",
        "would be a plus",
    ),
    "soft_skills": (
        "soft skills",
        "personal attributes",
        "competencies",
        "key attributes",
    ),
    "education": (
        "education",
        "education requirements",
        "academic requirements",
    ),
    "certifications": (
        "certifications",
        "certifications required",
        "licenses",
    ),
    # Content that must never influence matching. Benefits, legal boilerplate,
    # company marketing and internal stack listings add hundreds of words of
    # unrelated vocabulary that drag whole-document semantic similarity down -
    # and, worse, the tools named in them were being read as things the
    # candidate is required to have. A company saying its platform runs on
    # Kubernetes is describing itself, not setting a hiring bar.
    "boilerplate": (
        "about us",
        "about the company",
        "about our company",
        "who we are",
        "our company",
        "company overview",
        "company profile",
        "our story",
        "our mission",
        "our values",
        "culture",
        # Internal stack and team blurbs.
        "tech stack",
        "our tech stack",
        "the tech stack",
        "technology stack",
        "our technology stack",
        "our stack",
        "tools we use",
        "technologies we use",
        "what we build",
        "what we do",
        "our product",
        "our platform",
        "the team",
        "about the team",
        "meet the team",
        "your team",
        "who you will work with",
        "who you'll work with",
        "why join us",
        "why work with us",
        "life at",
        "what we offer you",
        # Process and logistics.
        "hiring process",
        "interview process",
        "recruitment process",
        "next steps",
        "how we hire",
        "job type",
        "employment type",
        "working hours",
        "reports to",
        "benefits",
        "what we offer",
        "perks",
        "perks and benefits",
        "compensation",
        "salary",
        "salary range",
        "pay range",
        "equal opportunity",
        "equal opportunity employer",
        "eeo statement",
        "diversity and inclusion",
        "how to apply",
        "application process",
        "note",
        "disclaimer",
        "legal",
        "privacy",
    ),
}

JOB_SECTION_ORDER = SECTION_ORDER + ["preferred_qualifications", "boilerplate"]


@dataclass(frozen=True)
class SectionBundle:
    sections: dict[str, str]
    order: list[str]


def normalize_section_heading(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[\u2022•*:-]+$", "", value)
    value = re.sub(r"[^a-z0-9\s&/+-]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _alias_maps(document_type: str) -> list[dict[str, tuple[str, ...]]]:
    """Alias maps to try, most specific first."""
    if document_type == "job":
        return [JOB_SECTION_ALIASES, SECTION_ALIASES]
    return [SECTION_ALIASES]


def _detect_section_heading(line: str, document_type: str = "resume") -> str | None:
    normalized = normalize_section_heading(line)
    if not normalized:
        return None

    # A heading is a short standalone line. Requiring this prevents a sentence
    # such as "we are looking for someone with strong requirements gathering
    # skills" from being mistaken for a "requirements" heading.
    word_count = len(normalized.split())
    if word_count > 6:
        return None

    for alias_map in _alias_maps(document_type):
        for section_name, aliases in alias_map.items():
            for alias in aliases:
                alias_normalized = normalize_section_heading(alias)
                if not alias_normalized:
                    continue
                if normalized == alias_normalized:
                    return section_name
                # Tolerate trailing decoration such as "Responsibilities (Key)"
                # or "Requirements -" without swallowing longer sentences.
                if normalized.startswith(alias_normalized) and len(normalized) <= len(alias_normalized) + 8:
                    return section_name
    return None


def split_preserving_blocks(text: str) -> list[str]:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.replace("\r", "\n").split("\n")]
    return [line for line in lines if line]


def extract_sections(text: str, document_type: str = "resume") -> SectionBundle:
    """Split a document into canonical sections.

    ``document_type="job"`` additionally recognises job-description headings
    (Responsibilities, Requirements, Benefits, ...) and maps them onto the same
    canonical keys used for resumes. The default keeps the original behaviour
    for existing callers.
    """
    order = JOB_SECTION_ORDER if document_type == "job" else SECTION_ORDER
    lines = split_preserving_blocks(text)
    sections: dict[str, list[str]] = {key: [] for key in order}
    current_section = "professional_summary"
    preamble: list[str] = []
    seen_section = False

    for line in lines:
        heading = _detect_section_heading(line, document_type=document_type)
        if heading:
            current_section = heading
            seen_section = True
            continue

        if not seen_section and current_section == "professional_summary":
            preamble.append(line)
            continue

        sections.setdefault(current_section, []).append(line)

    if preamble:
        sections["professional_summary"] = preamble + sections["professional_summary"]

    return SectionBundle(
        sections={key: "\n".join(value).strip() for key, value in sections.items()},
        order=order,
    )


def meaningful_text(bundle: SectionBundle, exclude: set[str] | None = None) -> str:
    """All section text except boilerplate, and anything else the caller drops.

    Benefits and equal-opportunity statements are genuine text but carry no
    matching signal, and including them dilutes whole-document similarity.
    Callers comparing a resume against a posting also exclude the preferred
    block, so optional extras cannot show up as missing keywords.
    """
    skip = {"boilerplate"} | (exclude or set())
    parts = [
        text
        for key, text in bundle.sections.items()
        if text and key not in skip
    ]
    return "\n".join(parts).strip()


def sentence_chunks(text: str) -> list[str]:
    blocks = split_preserving_blocks(text)
    chunks: list[str] = []
    for block in blocks:
        split_block = re.split(r"(?<=[.!?])\s+|\s*[•\-–]\s+", block)
        for sentence in split_block:
            sentence = sentence.strip(" -•\t")
            if sentence:
                chunks.append(sentence)
    return chunks


def extract_bullet_items(text: str) -> list[str]:
    bullets: list[str] = []
    for block in split_preserving_blocks(text):
        if re.match(r"^[•\-*–]\s+", block):
            bullets.append(re.sub(r"^[•\-*–]\s+", "", block).strip())
        else:
            bullets.append(block)
    return [item for item in bullets if item]


def section_text(bundle: SectionBundle, section_name: str) -> str:
    return bundle.sections.get(section_name, "")