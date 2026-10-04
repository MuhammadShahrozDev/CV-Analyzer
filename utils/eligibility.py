"""Eligibility screening.

This module answers the questions a recruiter asks *before* looking at an ATS
score: can this person legally and practically take the job, and do they clear
the stated bar for experience and education?

Nothing here feeds the ATS score. Location, work mode, years of experience and
degree level are hard filters in a real hiring process, not quality signals -
mixing them into a match percentage produces a number that means neither one
thing nor the other. The screening result is returned alongside the score and
displayed as its own section.

The module is deliberately dependency-free (standard library only), so it stays
fast and cannot fail because an optional model is missing.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Status vocabulary
# ---------------------------------------------------------------------------
MATCH = "match"          # clears the requirement
BORDERLINE = "borderline"  # close enough that a recruiter should decide
FAIL = "fail"            # clearly does not clear it
UNKNOWN = "unknown"      # not enough information in either document

_STATUS_RANK = {MATCH: 0, UNKNOWN: 1, BORDERLINE: 2, FAIL: 3}


# ---------------------------------------------------------------------------
# Geography
#
# A curated list beats a geocoding dependency here: job postings and resumes
# name a small, predictable set of places, and an offline lookup cannot fail at
# request time. Ambiguous names that exist in more than one country (Hyderabad,
# Cambridge, Birmingham) are deliberately omitted rather than guessed.
# ---------------------------------------------------------------------------
_CITY_COUNTRY: dict[str, str] = {
    # Pakistan
    "lahore": "pakistan", "karachi": "pakistan", "islamabad": "pakistan",
    "rawalpindi": "pakistan", "faisalabad": "pakistan", "multan": "pakistan",
    "peshawar": "pakistan", "quetta": "pakistan", "sialkot": "pakistan",
    "gujranwala": "pakistan", "abbottabad": "pakistan", "bahawalpur": "pakistan",
    # India
    "bangalore": "india", "bengaluru": "india", "mumbai": "india",
    "new delhi": "india", "gurgaon": "india", "gurugram": "india",
    "noida": "india", "chennai": "india", "pune": "india", "kolkata": "india",
    "ahmedabad": "india",
    # United States
    "new york": "united states", "san francisco": "united states",
    "seattle": "united states", "austin": "united states",
    "boston": "united states", "chicago": "united states",
    "los angeles": "united states", "atlanta": "united states",
    "denver": "united states", "dallas": "united states",
    "houston": "united states", "san jose": "united states",
    "washington dc": "united states", "miami": "united states",
    # United Kingdom & Ireland
    "london": "united kingdom", "manchester": "united kingdom",
    "leeds": "united kingdom", "bristol": "united kingdom",
    "glasgow": "united kingdom", "edinburgh": "united kingdom",
    "dublin": "ireland",
    # Europe
    "berlin": "germany", "munich": "germany", "hamburg": "germany",
    "frankfurt": "germany", "amsterdam": "netherlands",
    "rotterdam": "netherlands", "paris": "france", "madrid": "spain",
    "barcelona": "spain", "lisbon": "portugal", "milan": "italy",
    "rome": "italy", "warsaw": "poland", "krakow": "poland",
    "stockholm": "sweden", "copenhagen": "denmark", "oslo": "norway",
    "helsinki": "finland", "zurich": "switzerland", "vienna": "austria",
    "brussels": "belgium", "prague": "czech republic", "budapest": "hungary",
    # Middle East
    "dubai": "united arab emirates", "abu dhabi": "united arab emirates",
    "sharjah": "united arab emirates", "doha": "qatar", "riyadh": "saudi arabia",
    "jeddah": "saudi arabia", "manama": "bahrain", "kuwait city": "kuwait",
    "muscat": "oman", "amman": "jordan", "cairo": "egypt",
    # Asia Pacific
    "singapore": "singapore", "kuala lumpur": "malaysia", "jakarta": "indonesia",
    "manila": "philippines", "bangkok": "thailand", "hanoi": "vietnam",
    "ho chi minh city": "vietnam", "tokyo": "japan", "osaka": "japan",
    "seoul": "south korea", "hong kong": "hong kong", "shanghai": "china",
    "beijing": "china", "shenzhen": "china",
    "sydney": "australia", "melbourne": "australia", "brisbane": "australia",
    "perth": "australia", "auckland": "new zealand", "wellington": "new zealand",
    # Americas (non-US)
    "toronto": "canada", "vancouver": "canada", "montreal": "canada",
    "calgary": "canada", "ottawa": "canada",
    "mexico city": "mexico", "sao paulo": "brazil", "buenos aires": "argentina",
    # Africa
    "lagos": "nigeria", "abuja": "nigeria", "nairobi": "kenya",
    "cape town": "south africa", "johannesburg": "south africa",
}

_COUNTRY_ALIASES: dict[str, str] = {
    "usa": "united states", "u s a": "united states", "us": "united states",
    "united states of america": "united states", "america": "united states",
    "uk": "united kingdom", "u k": "united kingdom",
    "great britain": "united kingdom", "britain": "united kingdom",
    "england": "united kingdom", "scotland": "united kingdom",
    "wales": "united kingdom",
    "uae": "united arab emirates", "u a e": "united arab emirates",
    "ksa": "saudi arabia", "holland": "netherlands", "deutschland": "germany",
    "pk": "pakistan", "roi": "ireland",
}

_COUNTRIES: set[str] = set(_CITY_COUNTRY.values()) | {
    "argentina", "bahrain", "bangladesh", "belgium", "brazil", "canada",
    "china", "czech republic", "denmark", "egypt", "finland", "france",
    "germany", "greece", "hong kong", "hungary", "india", "indonesia",
    "ireland", "italy", "japan", "jordan", "kenya", "kuwait", "malaysia",
    "mexico", "nepal", "netherlands", "new zealand", "nigeria", "norway",
    "oman", "pakistan", "philippines", "poland", "portugal", "qatar",
    "romania", "saudi arabia", "singapore", "south africa", "south korea",
    "spain", "sri lanka", "sweden", "switzerland", "thailand", "turkey",
    "ukraine", "united arab emirates", "united kingdom", "united states",
    "vietnam", "austria",
}

# Phrases that mean "location does not constrain this hire".
_GLOBAL_REMOTE_PATTERNS = (
    r"remote\s*[-,(]?\s*(?:worldwide|global|anywhere|international)",
    r"(?:work|hire)\s+from\s+anywhere",
    r"fully\s+remote",
    r"100%\s*remote",
    r"remote\s+first",
)

_REMOTE_PATTERN = re.compile(r"\bremote(?:ly)?\b|\bwork\s+from\s+home\b|\bwfh\b", re.I)
_HYBRID_PATTERN = re.compile(r"\bhybrid\b|\bpartly\s+remote\b|\d\s*days?\s+(?:in\s+(?:the\s+)?office|on\s*-?\s*site)", re.I)
_ONSITE_PATTERN = re.compile(r"\bon\s*-?\s*site\b|\bonsite\b|\bin\s*-?\s*office\b|\bin\s+person\b|\bfully\s+office\s+based\b", re.I)

_LOCATION_LABEL = re.compile(
    r"^\s*(?:job\s+)?(?:location|based\s+in|office|city|work\s+location|address)\s*[:\-\u2013]\s*(.+)$",
    re.I,
)


def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s,./+#-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _canonical_country(value: str) -> str:
    value = _normalize(value)
    value = _COUNTRY_ALIASES.get(value, value)
    return value if value in _COUNTRIES else ""


def _lines(text: str) -> list[str]:
    return [line.strip() for line in (text or "").replace("\r", "\n").split("\n") if line.strip()]


def _find_place(fragment: str) -> tuple[str, str]:
    """Return (city, country) found anywhere in a fragment. Either may be blank."""
    normalized = _normalize(fragment)
    if not normalized:
        return "", ""

    city = ""
    country = ""

    for name, owning_country in _CITY_COUNTRY.items():
        if re.search(rf"\b{re.escape(name)}\b", normalized):
            city = name
            country = owning_country
            break

    for name in sorted(_COUNTRIES | set(_COUNTRY_ALIASES), key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", normalized):
            resolved = _canonical_country(name)
            if resolved:
                country = resolved
            break

    return city, country


def extract_location(text: str, *, header_lines: int = 0) -> dict[str, str]:
    """Best-effort location for a document.

    ``header_lines`` restricts the first pass to the top of the document, which
    is where a resume states the candidate's own city. Without that restriction
    a resume mentioning a former employer's head office would be read as the
    candidate living there.
    """
    lines = _lines(text)
    scan = lines[:header_lines] if header_lines else lines

    # 1. An explicit label is the most reliable signal in either document.
    for line in scan or lines:
        labelled = _LOCATION_LABEL.match(line)
        if labelled:
            city, country = _find_place(labelled.group(1))
            if city or country:
                return {"city": city, "country": country, "raw": labelled.group(1).strip()}

    # 2. Otherwise look for a recognised place name in the scanned region.
    for line in scan:
        city, country = _find_place(line)
        if city or country:
            return {"city": city, "country": country, "raw": line.strip()}

    # 3. Fall back to the whole document.
    if header_lines:
        for line in lines:
            city, country = _find_place(line)
            if city or country:
                return {"city": city, "country": country, "raw": line.strip()}

    return {"city": "", "country": "", "raw": ""}


def extract_work_mode(job_text: str) -> str:
    """Return 'remote', 'remote-global', 'hybrid', 'onsite' or 'unknown'."""
    text = job_text or ""
    for pattern in _GLOBAL_REMOTE_PATTERNS:
        if re.search(pattern, text, re.I):
            return "remote-global"
    if _HYBRID_PATTERN.search(text):
        return "hybrid"
    if _REMOTE_PATTERN.search(text):
        return "remote"
    if _ONSITE_PATTERN.search(text):
        return "onsite"
    return "unknown"


def _experience_rows_from(analysis: dict[str, Any]) -> list[dict[str, str]]:
    """The three card rows, taken straight from the normalized result."""
    return [
        {"label": "Required Experience", "value": analysis["required_label"]},
        {"label": "Detected Experience", "value": analysis["detected_label"]},
        {"label": "Experience Gap", "value": analysis["gap_label"]},
    ]


def _experience_rows(
    candidate_years: float | None,
    required_years: float | None,
    requirement: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Required / Detected / Gap, which is how a recruiter reads experience.

    "Resume: 0.1 years / Job: 2 years" makes the reader do the subtraction.
    The gap is the number the decision actually turns on, so it is stated.
    """
    # Single source of truth: the card shows exactly what the extractor found,
    # including a range ("3-5 years") or a preferred-only figure. It can no
    # longer read "Not stated" while the backend holds a requirement.
    required = (
        format_required_experience(requirement)
        if requirement is not None
        else (_years_phrase(required_years) if required_years is not None else "Not stated")
    )
    detected = _years_phrase(candidate_years) if candidate_years is not None else "Not stated"

    if required_years is None or candidate_years is None:
        gap = "Cannot be calculated"
    elif candidate_years >= required_years:
        surplus = candidate_years - required_years
        gap = "None" if surplus <= 0 else f"None ({_years_phrase(surplus)} above)"
    else:
        gap = f"{_years_phrase(round(required_years - candidate_years, 1))} short"

    return [
        {"label": "Required Experience", "value": required},
        {"label": "Detected Experience", "value": detected},
        {"label": "Experience Gap", "value": gap},
    ]


def _years_phrase(value: float) -> str:
    """'1 year' / '3 years' / '2.5 years' - never 'year(s)'."""
    return f"{value:g} year" if value == 1 else f"{value:g} years"


def _title(value: str) -> str:
    return " ".join(part.capitalize() for part in value.split()) if value else ""


def format_location(place: dict[str, str]) -> str:
    city = _title(place.get("city", ""))
    country = _title(place.get("country", ""))
    if city and country:
        return f"{city}, {country}"
    return city or country or (place.get("raw") or "Not stated")


def check_location(candidate: dict[str, str], job: dict[str, str], work_mode: str) -> dict[str, Any]:
    """Decide whether the candidate can work where the job is."""
    candidate_city = candidate.get("city", "")
    candidate_country = candidate.get("country", "")
    job_city = job.get("city", "")
    job_country = job.get("country", "")

    if work_mode == "remote-global":
        return {
            "status": MATCH,
            "detail": "Role is fully remote, so candidate location does not restrict eligibility.",
        }

    if work_mode == "remote" and not (job_city or job_country):
        return {"status": MATCH, "detail": "Remote role with no stated location restriction."}

    if not (job_city or job_country):
        return {"status": UNKNOWN, "detail": "The posting does not state a work location."}

    if not (candidate_city or candidate_country):
        return {"status": UNKNOWN, "detail": "The resume does not state a location."}

    same_city = bool(candidate_city and job_city and candidate_city == job_city)
    same_country = bool(candidate_country and job_country and candidate_country == job_country)

    if same_city:
        return {"status": MATCH, "detail": f"Candidate is based in {_title(job_city)}, the same city as the role."}

    if work_mode == "remote":
        if same_country or not job_country:
            return {"status": MATCH, "detail": "Remote role and the candidate is within the stated region."}
        return {
            "status": BORDERLINE,
            "detail": (
                f"Remote role restricted to {_title(job_country)}; the candidate is based in "
                f"{_title(candidate_country) or 'another country'}. "
                "Confirm right to work and time-zone overlap."
            ),
        }

    if same_country:
        return {
            "status": BORDERLINE,
            "detail": (
                f"Candidate is in {_title(job_country)} but not in "
                + (_title(job_city) if job_city else "the role's city")
                + ". The role would require relocation or a commute."
            ),
        }

    mode_label = "hybrid" if work_mode == "hybrid" else "on-site"
    return {
        "status": FAIL,
        "detail": (
            f"Role is {mode_label} in {format_location(job)}; "
            f"the candidate is based in {format_location(candidate)}."
        ),
    }


# ---------------------------------------------------------------------------
# Experience
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Required experience, read from the job description
#
# The previous pattern required the number to sit directly against the word
# "years", allowing only "+", "plus" or a range separator in between. Real
# postings do not co-operate: "4 or more years of experience" puts two words
# there, so the requirement was never found and the card read "Not stated"
# while an explicit requirement sat in the text.
#
# The rules below are wording-driven rather than format-driven, and never scan
# the document for loose numbers. A figure only counts when it is a year
# figure AND it is tied to experience - by the wording of the phrase itself,
# by an experience word nearby, or by the heading of the block it sits under.
# That keeps salary figures, funding rounds, percentages, meeting times and
# office-day counts out of the result.
# ---------------------------------------------------------------------------

# Written-out numbers, so "four or more years of experience" reads the same as
# "4 or more years of experience".
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}

# Word-anchored on both sides, and never preceded by a digit, a decimal point
# or a currency symbol.
#
# Without these guards the alternation matched *inside* longer tokens: "125
# years" yielded 25, "$75M ... years" yielded 5, and the spelled-out numbers
# leaked into ordinary words - "written years" matched "ten years", "someone
# years" matched "one years". Any of those could outrank the real requirement,
# because the lowest mandatory figure found in the posting wins.
#
# The optional decimal also lets "2.5 years" read as 2.5 rather than 2.
_NUM = (
    r"(?<![\w.$\u00a3\u20ac])(?:\d{1,2}(?:\.\d)?|"
    + "|".join(_WORD_NUMBERS)
    + r")(?![\w.]|\s*[%$])"
)


def _as_years(token: str) -> float | None:
    """Turn '4' or 'four' into a number of years."""
    token = (token or "").strip().lower()
    if token.isdigit():
        return float(token)
    return float(_WORD_NUMBERS[token]) if token in _WORD_NUMBERS else None


# Words permitted between the number and "years": "4 or more years", "4+ years".
_MORE = r"(?:\+|plus|or\s+(?:more|above|over|greater|higher)|or\s+longer)"
_YRS = r"(?:years?|yrs?)"

# Qualifiers introducing a minimum: "minimum of 4 years", "at least 3 years".
_MIN_LEAD = (
    r"(?:minimum|min\.?|at\s+least|no\s+less\s+than|not\s+less\s+than|"
    r"more\s+than|over|upwards?\s+of|in\s+excess\s+of)"
)

# Most specific first, so a range is read as a range before the loosest pattern
# can grab only its first number. Group 1 is always the minimum; group 2, where
# present, is the top of a stated range.
_REQUIREMENT_PATTERNS = (
    # "3-5 years", "3 to 5 years"
    re.compile(rf"({_NUM})\s*(?:-|\u2013|\u2014|to|through)\s*({_NUM})\s*\+?\s*{_YRS}", re.I),
    # "4 or more years", "four or more years", "4+ years", "4 plus years"
    re.compile(rf"({_NUM})\s*{_MORE}\s*{_YRS}", re.I),
    # "minimum of 4 years", "at least 3 years", "over 5 years"
    re.compile(rf"{_MIN_LEAD}\s+(?:of\s+)?({_NUM})\s*\+?\s*{_YRS}", re.I),
    # "experience of 4 years", "Experience: 2 years"
    re.compile(rf"experience\W{{0,4}}\s*(?:of\s+)?({_NUM})\s*\+?\s*{_YRS}", re.I),
    # "4 years of experience", and a bare "2 years" - loosest, applied last.
    re.compile(rf"({_NUM})\s*\+?\s*{_YRS}", re.I),
)

# Patterns whose own wording ties the figure to a requirement: a range, an
# explicit minimum, or the word "experience" inside the match itself. Only the
# bare "N years" pattern still needs an experience word nearby to be believed.
_SELF_EVIDENT_PATTERNS = {0, 1, 2, 3}

# What ties a year figure to *doing the work*. Postings often skip the noun
# entirely - "5 years building production software", "3 years leading teams" -
# so the activity verb has to count as an association in its own right.
_EXPERIENCE_WORD = re.compile(
    r"experien\w*|\bexp\b|background|track\s+record"
    r"|\b(?:work(?:ed|ing)?|build(?:ing)?|built|develop(?:ing|ed)?|design(?:ing|ed)?"
    r"|engineer(?:ing)?|program(?:ming)?|cod(?:ing|ed)?|writ(?:ing|ten)|ship(?:ping|ped)?"
    r"|deliver(?:ing|ed)?|lead(?:ing)?|led|manag(?:ing|ed)?|manage|manag(?:ement)?"
    r"|architect(?:ing)?|maintain(?:ing)?|support(?:ing)?|operat(?:ing)?)\b",
    re.I,
)

# Subjects other than the candidate. "Backed by founders with more than 10
# years of experience" describes the company, not a hiring bar, and reading it
# as one produced an impossible requirement.
_THIRD_PARTY_SUBJECT = re.compile(
    r"\b(?:founder|founders|co-?founder|ceo|cto|our\s+team|the\s+team|leadership|"
    r"executives?|advisors?|investors?|backers?|board|management\s+team|"
    r"backed\s+by|led\s+by|founded\s+by|combined|collective|between\s+them|"
    r"average|averaging|company|we\s+have|we've)\b",
    re.I,
)

# Qualifiers that keep a requirement general rather than tying it to one tool.
_GENERIC_EXPERIENCE_QUALIFIER = frozenset({
    "", "professional", "industry", "relevant", "work", "working", "domain",
    "hands on", "hands-on", "technical", "practical", "prior", "previous",
    "overall", "total", "combined", "full time", "full-time", "commercial",
    "software", "software engineering", "software development", "engineering",
    "development", "programming", "product", "production", "applicable",
    "related", "similar", "equivalent", "post qualification", "paid",
})

# Headings that open a block of hiring requirements. A year figure inside one
# is a requirement even without the word "experience" on the same line, which
# is what makes a bare bullet such as "2+ years" readable.
_REQUIREMENT_HEADING = re.compile(
    r"^(?:"
    r"(?:minimum\s+|basic\s+|key\s+|core\s+|essential\s+)?(?:requirements?|qualifications?)"
    r"|what\s+you'?l{1,2}\s+bring|what\s+we'?re\s+looking\s+for|what\s+you'?ll\s+need"
    r"|who\s+you\s+are|about\s+you|you\s+(?:have|will\s+have)|your\s+(?:background|profile)"
    r"|must[\s-]?haves?|skills?(?:\s*(?:&|and)\s*experience)?|experience"
    r"|the\s+ideal\s+candidate|we'?d\s+love\s+to\s+see"
    r")\s*:?\s*$",
    re.I,
)

_PREFERRED_HEADING = re.compile(
    r"^(?:preferred\s+(?:qualifications?|skills?|experience)|nice[\s-]to[\s-]haves?"
    r"|bonus(?:\s+points)?|desirable|good\s+to\s+have|pluses)\s*:?\s*$",
    re.I,
)

# Marks a line as a preference rather than a hard requirement.
_PREFERRED_MARKER = re.compile(
    r"\bpreferred\b|\bpreferable\b|\bpreferably\b|\bnice\s+to\s+have\b|\bdesirable\b"
    r"|\bbonus\b|\ba\s+plus\b|\badvantageous\b|\bideally\b|\bwould\s+be\s+great\b"
    r"|\bnot\s+required\b|\boptional\b",
    re.I,
)

# "no prior experience required", "experience is not required"
_NO_EXPERIENCE = re.compile(
    r"no\s+(?:minimum|prior|previous|professional|formal|work)?\s*(?:years?\s+of\s+)?experience\s*(?:is\s+)?"
    r"(?:required|necessary|needed|expected)?"
    r"|experience\s+(?:is\s+)?(?:not|isn't)\s+(?:required|necessary|needed)"
    r"|without\s+(?:any\s+)?(?:prior\s+)?experience",
    re.I,
)

# Phrases that use "years" for something other than a hiring bar.
#
# These are judged POSITIONALLY, in a window around the figure - never against
# the whole clause. A requirement bullet often runs on into unrelated prose
# ("4 or more years of experience building production software, maintaining
# contract integrations"), and vetoing on any occurrence anywhere meant one
# distant word deleted a real requirement and the card read "Not stated".
_NON_REQUIREMENT_CONTEXT = re.compile(
    r"\b(?:founded|established|incorporated|launched|since|ago|old|anniversary|"
    r"past|last|next|history|runway|contract|visa|tenure|every)\b",
    re.I,
)

# How much text either side of the figure the veto looks at. "over the past 5
# years" and "5 years ago" both sit well inside this; a clause tail thirty
# words later does not.
_VETO_BEFORE = 30
_VETO_AFTER = 14


def _vetoed(text: str, span: tuple[int, int]) -> bool:
    """Whether a year figure is being used for something other than a hiring bar."""
    before = text[max(0, span[0] - _VETO_BEFORE):span[0]]
    after = text[span[1]:span[1] + _VETO_AFTER]
    return bool(_NON_REQUIREMENT_CONTEXT.search(before) or _NON_REQUIREMENT_CONTEXT.search(after))

# How far a year phrase may sit from an experience word and still count.
_EXPERIENCE_PROXIMITY = 80

# Kept for the resume side, which is unchanged: a candidate stating "3 years of
# experience" writes it plainly, and that logic already worked.
_YEARS_REQUIRED = re.compile(
    r"(\d{1,2})\s*(?:\+|plus)?\s*(?:\u2013|-|to)?\s*(\d{1,2})?\s*\+?\s*(?:years?|yrs?)"
    r"(?:\s+(?:of|in|with|as|experience))?",
    re.I,
)
_EXPERIENCE_CONTEXT = re.compile(r"experience|exp\b|background|track record", re.I)

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_DATE_RANGE = re.compile(
    r"(?:(?P<m1>[a-z]{3,9})\.?\s*)?(?P<y1>(?:19|20)\d{2})\s*(?:\u2013|\u2014|-|to|until)\s*"
    r"(?:(?P<m2>[a-z]{3,9})\.?\s*)?(?P<y2>(?:19|20)\d{2}|present|current|now|date|today)",
    re.I,
)


def _requirement_lines(text: str):
    """Yield cleaned lines, with bullets and markdown emphasis removed.

    Formatting must never decide whether a requirement is found, so leading
    bullets, dashes, list markers and *bold*/_italic_ marks are stripped before
    anything is matched.
    """
    for raw_line in (text or "").replace("\r", "\n").split("\n"):
        line = re.sub(r"^[\s\-\u2013\u2022\u25cf\u25aa*+>#]+", "", raw_line.strip())
        line = re.sub(r"[*_`]+", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            yield line


def _near_experience(line: str, span: tuple[int, int]) -> bool:
    """Whether a year phrase sits close enough to an experience word to count."""
    for hit in _EXPERIENCE_WORD.finditer(line):
        if hit.start() < span[0]:
            distance = span[0] - hit.end()
        elif hit.start() > span[1]:
            distance = hit.start() - span[1]
        else:
            distance = 0
        if distance <= _EXPERIENCE_PROXIMITY:
            return True
    return False


# Clause boundaries. A bullet often carries a requirement and an aside in the
# same breath - "4 or more years of experience building production software,
# and we have shipped every week for the last 2 years". Judging such a line as
# one unit meant a single noise word ("last") discarded the whole thing,
# requirement included, and a larger figure elsewhere in the posting won by
# default. Clauses are judged separately so only the aside is dropped.
_CLAUSE_SPLIT = re.compile(r"(?<=[.;!?])\s+|,\s+(?=(?:and|but|or|while|where|plus)\b)|\s+[\u2013\u2014]\s+", re.I)


def _clauses(line: str) -> list[str]:
    parts = [part.strip() for part in _CLAUSE_SPLIT.split(line) if part and part.strip()]
    return parts or [line]


# "2+ years of React experience" names a tool; "5+ years of software
# engineering experience" does not. The distinction matters because the
# overall hiring bar is the general one - a tool-specific figure is almost
# always the smaller of the two, and letting it win understates the role.
_SKILL_SCOPED = re.compile(
    rf"{_YRS}\s+(?:of\s+|in\s+|with\s+)?([A-Za-z][\w.+#/ -]{{0,28}}?)\s+experien",
    re.I,
)
_SKILL_AFTER_EXPERIENCE = re.compile(
    rf"{_YRS}\s+of\s+experience\s+(?:with|in|using|building\s+with)\s+([A-Za-z][\w.+#/ -]{{0,28}})",
    re.I,
)


def _requirement_scope(clause: str, span: tuple[int, int]) -> tuple[str, str]:
    """Return (scope, subject) where scope is 'general' or 'skill'.

    Only the wording that immediately follows the year figure is examined, so a
    tool named elsewhere in the sentence cannot narrow the requirement.
    """
    tail = clause[span[0]:span[0] + 90]

    match = _SKILL_SCOPED.search(tail)
    if match:
        qualifier = re.sub(r"\s+", " ", match.group(1).strip().lower())
        if qualifier not in _GENERIC_EXPERIENCE_QUALIFIER:
            return "skill", match.group(1).strip()
        return "general", ""

    # "5+ years of experience with React and TypeScript" is a general
    # requirement that happens to mention tools, not a tool-specific one.
    if _SKILL_AFTER_EXPERIENCE.search(tail):
        return "general", ""

    return "general", ""


def _years_in_line(line: str, in_block: bool = False):
    """Every credible year requirement in a line, as (low, high, matched text).

    Rejection is judged per clause, association per line: a neighbouring aside
    can no longer veto a requirement, while a figure and the word "experience"
    still count as related when a comma separates them.
    """
    found: list[tuple[float, float | None, str, str, str]] = []

    for clause in _clauses(line):
        if _NO_EXPERIENCE.search(clause):
            continue

        # Every figure in the clause, not just the first. A single sentence can
        # carry two requirements - "2+ years of React experience and 5+ years
        # of software engineering experience" - and stopping at the first one
        # let the smaller, tool-specific figure stand in for the whole role.
        #
        # Patterns are ordered most specific first and accepted spans are
        # remembered, so a range is never counted a second time by the bare
        # pattern that would also match part of it.
        taken: list[tuple[int, int]] = []

        for index, pattern in enumerate(_REQUIREMENT_PATTERNS):
            for match in pattern.finditer(clause):
                span = match.span()
                if any(span[0] < end and start < span[1] for start, end in taken):
                    continue

                if _vetoed(clause, span):
                    continue

                # A figure describing the founders, the team or the company is
                # not a hiring bar for the candidate.
                if _THIRD_PARTY_SUBJECT.search(clause[max(0, span[0] - 60):span[0]]):
                    continue

                trusted = index in _SELF_EVIDENT_PATTERNS
                if not (trusted or in_block or _near_experience(line, span)):
                    continue

                low = _as_years(match.group(1))
                if low is None or not 0 < low <= 40:
                    continue

                high = None
                if pattern.groups >= 2 and match.lastindex and match.lastindex >= 2:
                    high = _as_years(match.group(2) or "")

                scope, subject = _requirement_scope(clause, span)
                found.append((low, high, match.group(0), scope, subject))
                taken.append(span)

    return found


# Last-resort pass. An explicit "N years of experience" phrase is the least
# ambiguous statement a posting can make, so it must never be lost to heading
# detection, clause splitting or a veto. If the structured pass finds nothing,
# this looks for that phrase alone and takes the lowest value it states.
_EXPLICIT_EXPERIENCE = re.compile(
    rf"({_NUM})\s*(?:{_MORE})?\s*{_YRS}(?:[^.;!?]{{0,40}}?)\bexperien\w*",
    re.I,
)
_EXPLICIT_EXPERIENCE_REVERSED = re.compile(
    rf"\bexperien\w*(?:[^.;!?]{{0,25}}?)({_NUM})\s*(?:{_MORE})?\s*{_YRS}",
    re.I,
)


def _fallback_experience(job_text: str) -> list[tuple[float, float | None, str, str, str]]:
    """Explicit 'N years ... experience' phrases, found without any structure."""
    found: list[tuple[float, float | None, str, str, str]] = []
    for line in _requirement_lines(job_text):
        if _NO_EXPERIENCE.search(line):
            continue
        for pattern in (_EXPLICIT_EXPERIENCE, _EXPLICIT_EXPERIENCE_REVERSED):
            for match in pattern.finditer(line):
                if _vetoed(line, match.span(1)):
                    continue
                if _THIRD_PARTY_SUBJECT.search(line[max(0, match.start() - 60):match.start()]):
                    continue
                years = _as_years(match.group(1))
                if years is not None and 0 < years <= 40:
                    scope, subject = _requirement_scope(line, match.span(1))
                    found.append((years, None, match.group(0)[:60], scope, subject))
    return found


def extract_experience_requirement(job_text: str) -> dict[str, Any]:
    """The experience bar a posting sets, separating required from preferred.

    Returns min_years (the mandatory minimum, or None), max_years (the top of a
    stated range), preferred_years (a figure the posting only prefers), and the
    text the figure was read from.
    """
    required: list[tuple[float, float | None, str, str, str]] = []
    preferred: list[tuple[float, float | None, str, str, str]] = []
    in_requirement_block = False
    in_preferred_block = False

    for line in _requirement_lines(job_text):
        # Headings switch which bucket following lines belong to, so a figure
        # under "Preferred Qualifications" is never read as mandatory.
        if _PREFERRED_HEADING.match(line):
            in_preferred_block, in_requirement_block = True, False
            continue
        if _REQUIREMENT_HEADING.match(line):
            in_requirement_block, in_preferred_block = True, False
            continue

        # Rejection now happens per clause inside _years_in_line, so a line is
        # never discarded wholesale because one of its clauses is noise.
        for entry in _years_in_line(line, in_requirement_block or in_preferred_block):
            is_preferred = in_preferred_block or bool(_PREFERRED_MARKER.search(line))
            (preferred if is_preferred else required).append(entry)

    # Safety net: never report "Not stated" while an explicit "N years of
    # experience" phrase sits in the text.
    if not required and not preferred:
        required = _fallback_experience(job_text)

    def describe(entries):
        return [
            {"years": low, "max_years": high, "text": raw, "scope": scope, "subject": subject}
            for low, high, raw, scope, subject in entries
        ]

    # Skill-scoped figures ("2+ years of React experience") are kept out of the
    # overall bar and reported separately, so a tool-specific number cannot
    # understate a role that also asks for general engineering experience.
    skill_specific = [entry for entry in required if entry[3] == "skill"]
    general = [entry for entry in required if entry[3] != "skill"] or skill_specific

    diagnostics = {
        "required_candidates": describe(required),
        "preferred_candidates": describe(preferred),
        "skill_requirements": [
            {"skill": subject, "years": low, "text": raw}
            for low, _high, raw, scope, subject in required
            if scope == "skill" and subject
        ],
    }

    if general:
        low, high, raw, _scope, _subject = min(general, key=lambda item: item[0])
        return {"min_years": low, "max_years": high, "preferred_years": None,
                "raw": raw, **diagnostics}
    if preferred:
        low, high, raw, _scope, _subject = min(preferred, key=lambda item: item[0])
        return {"min_years": None, "max_years": high, "preferred_years": low,
                "raw": raw, **diagnostics}
    return {"min_years": None, "max_years": None, "preferred_years": None,
            "raw": "", **diagnostics}


# Status vocabulary for the normalized result (Part 17).
EXPERIENCE_MATCH = "match"
EXPERIENCE_BORDERLINE = "borderline"
EXPERIENCE_BELOW = "below_requirement"
EXPERIENCE_NOT_STATED = "not_stated"


def analyze_experience(
    job_text: str,
    resume_text: str,
    resume_experience_text: str = "",
) -> dict[str, Any]:
    """The one normalized experience result the whole application consumes.

    Every consumer - the eligibility card, the recruiter decision, the
    feedback, the recommendations - reads from this single structure, so no two
    parts of the UI can disagree about what the posting asks for.

    Keys:
        required_experience  minimum years the posting mandates, or None
        minimum_experience   same value, named for the range case
        preferred_experience top of a stated range ("3-5 years" -> 5), or None
        detected_experience  years read from the resume, or None
        experience_gap       positive number of years short, or None
        surplus_experience   years above the bar when the candidate clears it
        status               match | borderline | below_requirement | not_stated
        source_text          the phrase the requirement was read from
        skill_requirements   tool-specific figures, kept out of the overall bar
    """
    requirement = extract_experience_requirement(job_text)
    candidate = extract_candidate_experience(resume_text, resume_experience_text)

    required = requirement["min_years"]
    detected = candidate["years"]
    verdict = check_experience(detected, required)

    gap = None
    surplus = None
    if required is not None and detected is not None:
        difference = round(detected - required, 1)
        if difference < 0:
            gap = abs(difference)
        else:
            surplus = difference

    if required is None:
        status = EXPERIENCE_NOT_STATED
    elif verdict["status"] == MATCH:
        status = EXPERIENCE_MATCH
    elif verdict["status"] == BORDERLINE:
        status = EXPERIENCE_BORDERLINE
    elif verdict["status"] == FAIL:
        status = EXPERIENCE_BELOW
    else:
        status = EXPERIENCE_NOT_STATED

    return {
        "required_experience": required,
        "minimum_experience": required,
        "preferred_experience": requirement["max_years"],
        "preferred_only_experience": requirement["preferred_years"],
        "detected_experience": detected,
        "detected_source": candidate["source"],
        "experience_gap": gap,
        "surplus_experience": surplus,
        "status": status,
        "verdict": verdict,
        "source_text": requirement["raw"],
        "skill_requirements": requirement.get("skill_requirements", []),
        "required_label": format_required_experience(requirement),
        "detected_label": _years_phrase(detected) if detected is not None else "Not stated",
        "gap_label": _experience_gap_label(required, detected),
        "requirement": requirement,
    }


def _experience_gap_label(required: float | None, detected: float | None) -> str:
    """Human-readable gap. A surplus is never shown as a negative shortfall."""
    if required is None or detected is None:
        return "Cannot be calculated"
    difference = round(detected - required, 1)
    if difference >= 0:
        return "None" if difference == 0 else f"None ({_years_phrase(difference)} above)"
    return f"{_years_phrase(abs(difference))} short"


def format_required_experience(requirement: dict[str, Any]) -> str:
    """How the requirement reads on the card: '4+ years', '3-5 years'."""
    minimum = requirement.get("min_years")
    maximum = requirement.get("max_years")
    if minimum is None:
        preferred = requirement.get("preferred_years")
        return f"{preferred:g}+ years (preferred)" if preferred is not None else "Not stated"
    if maximum is not None and maximum > minimum:
        return f"{minimum:g}-{maximum:g} years"
    return f"{minimum:g}+ years"


def extract_required_experience(job_text: str) -> float | None:
    """Minimum years the posting mandates, or None if it never says.

    Thin wrapper over extract_experience_requirement, kept so existing callers
    and tests keep working unchanged.
    """
    return extract_experience_requirement(job_text)["min_years"]


def _month_index(value: str | None) -> int:
    if not value:
        return 1
    return _MONTHS.get(value[:3].lower(), 1)


def _stated_candidate_years(resume_text: str) -> float | None:
    values: list[float] = []
    for line in _lines(resume_text):
        if not _EXPERIENCE_CONTEXT.search(line):
            continue
        for match in _YEARS_REQUIRED.finditer(line):
            value = float(match.group(1))
            if 0 < value <= 40:
                values.append(value)
    return max(values) if values else None


def _years_from_date_ranges(text: str, today: date | None = None) -> float:
    """Total months covered by employment date ranges, overlaps merged."""
    today = today or date.today()
    spans: list[tuple[int, int]] = []

    for match in _DATE_RANGE.finditer(text or ""):
        start_year = int(match.group("y1"))
        start = start_year * 12 + _month_index(match.group("m1"))

        end_token = match.group("y2").lower()
        if end_token.isdigit():
            end = int(end_token) * 12 + _month_index(match.group("m2"))
        else:
            end = today.year * 12 + today.month

        if end < start or start_year < 1960:
            continue
        spans.append((start, end))

    if not spans:
        return 0.0

    spans.sort()
    merged: list[list[int]] = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    months = sum(end - start for start, end in merged)
    return round(months / 12.0, 1)


def extract_candidate_experience(resume_text: str, experience_text: str = "") -> dict[str, Any]:
    """Years of experience claimed or demonstrated by the resume."""
    stated = _stated_candidate_years(resume_text)
    derived = _years_from_date_ranges(experience_text or resume_text)

    if stated is not None and derived:
        # The candidate's own stated total wins. Dated roles overlap with
        # education, internships and side projects, so summing them tends to
        # overstate professional experience - and the stated figure is what a
        # recruiter reads off the summary line anyway.
        years = stated
        source = (
            "stated in the resume"
            if abs(derived - stated) < 1.0
            else f"stated in the resume (dated roles span about {_years_phrase(derived)})"
        )
    elif stated is not None:
        years = stated
        source = "stated in the resume"
    elif derived:
        years = derived
        source = "derived from dated roles"
    else:
        return {"years": None, "source": "", "detail": "The resume does not state a total length of experience."}

    return {"years": round(years, 1), "source": source, "detail": ""}


def check_experience(candidate_years: float | None, required_years: float | None) -> dict[str, Any]:
    """Compare experience without ever auto-rejecting.

    A year or two short of a posted figure is routinely interviewed, so a
    shortfall is reported as Borderline or Below Requirement and left for a
    person to weigh - never as a hard fail.
    """
    if required_years is None:
        if candidate_years is None:
            return {"status": UNKNOWN, "detail": "Neither document states a length of experience."}
        return {
            "status": MATCH,
            "detail": f"The posting sets no minimum; the candidate shows about {_years_phrase(candidate_years)}.",
        }

    if candidate_years is None:
        return {
            "status": UNKNOWN,
            "detail": f"The posting asks for {required_years:g}+ years; the resume does not state a total.",
        }

    if candidate_years >= required_years:
        return {
            "status": MATCH,
            "detail": f"{_years_phrase(candidate_years)} against a requirement of {required_years:g}+ years.",
        }

    gap = required_years - candidate_years
    if gap <= 1.0 or candidate_years >= 0.7 * required_years:
        return {
            "status": BORDERLINE,
            "detail": (
                f"{_years_phrase(gap)} below the stated requirement of {required_years:g}+ years, "
                "which is within the range normally considered."
            ),
        }

    return {
        "status": FAIL,
        "detail": (
            f"{_years_phrase(candidate_years)} against a requirement of {required_years:g}+ years, "
            f"a shortfall of {_years_phrase(gap)}."
        ),
        "label": "Below Requirement",
    }


# ---------------------------------------------------------------------------
# Education
# ---------------------------------------------------------------------------
DEGREE_LEVELS: dict[str, int] = {
    "doctorate": 5, "master": 4, "bachelor": 3, "associate": 2, "diploma": 2,
    "intermediate": 1,
}

_DEGREE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:ph\.?\s?d|doctorate|doctoral|d\.?phil)\b", "doctorate"),
    (r"\b(?:m\.?s\.?c?|master'?s?|mba|m\.?tech|m\.?eng|mca|mcs|m\.?phil|m\.?a\b|llm|ms)\b", "master"),
    (r"\b(?:b\.?s\.?c?|bachelor'?s?|b\.?tech|b\.?eng|bba|bca|bcs|b\.?a\b|b\.?ed|llb|be\b|bs)\b", "bachelor"),
    (r"\bbs(?:cs|se|it|ds|ai|dsa)\b", "bachelor"),
    (r"\b(?:associate'?s?\s+degree|associate\s+degree)\b", "associate"),
    (r"\b(?:diploma|hnd|higher\s+national)\b", "diploma"),
    (r"\b(?:intermediate|f\.?sc|ics|a\s?levels?|high\s+school)\b", "intermediate"),
)

# Abbreviated Pakistani/South Asian degree codes carry their field in the code
# itself ("BSDS" is a BS in Data Science), so they are expanded before the
# generic field scan runs.
_DEGREE_CODE_FIELDS: dict[str, str] = {
    "bscs": "computer science", "bsse": "software engineering",
    "bsit": "information technology", "bsds": "data science",
    "bsai": "artificial intelligence", "bsce": "computer engineering",
    "bba": "business administration", "mba": "business administration",
    "bca": "computer applications", "mca": "computer applications",
    "mcs": "computer science", "bcs": "computer science",
}

# Fields treated as interchangeable. A posting asking for "Computer Science or
# a related field" is satisfied by any member of the same group - which is the
# whole point of the phrase, and what a human screener does automatically.
FIELD_GROUPS: dict[str, tuple[str, ...]] = {
    "computing": (
        "computer science", "computer sciences", "computing", "data science",
        "software engineering", "software development", "information technology",
        "information systems", "artificial intelligence", "machine learning",
        "computer engineering", "data analytics", "data analysis",
        "business analytics", "statistics", "applied statistics", "mathematics",
        "applied mathematics", "computer applications", "informatics",
        "data engineering", "cyber security", "cybersecurity",
        "information security", "computational science", "quantitative",
    ),
    "engineering": (
        "engineering", "electrical engineering", "electronics engineering",
        "mechanical engineering", "civil engineering", "chemical engineering",
        "industrial engineering", "mechatronics", "telecommunications",
    ),
    "business": (
        "business", "business administration", "management", "commerce",
        "finance", "accounting", "economics", "marketing", "supply chain",
        "human resources",
    ),
    "design": (
        "design", "graphic design", "visual communication", "fine arts",
        "user experience", "interaction design", "multimedia",
    ),
    "science": (
        "physics", "chemistry", "biology", "biotechnology", "life sciences",
        "natural sciences", "environmental science",
    ),
}

_ALL_FIELDS: list[str] = sorted(
    {field for fields in FIELD_GROUPS.values() for field in fields},
    key=len,
    reverse=True,
)

_FIELD_GROUP_LOOKUP: dict[str, str] = {
    field: group for group, fields in FIELD_GROUPS.items() for field in fields
}

_RELATED_FIELD_PHRASE = re.compile(
    r"or\s+(?:a\s+)?(?:related|similar|relevant|equivalent)\s+(?:field|discipline|subject|degree|area)",
    re.I,
)
_EDUCATION_CONTEXT = re.compile(
    r"degree|bachelor|master|phd|doctorate|education|graduate|university|college|diploma"
    r"|b\.?s\.?c?\b|m\.?s\.?c?\b|bs\b|ms\b"
    r"|bs(?:cs|se|it|ds|ai|ce)\b|mcs\b|bcs\b|bca\b|mca\b|bba\b|mba\b|b\.?tech\b|m\.?tech\b"
    # Arts, engineering and law abbreviations. _DEGREE_PATTERNS below already
    # recognised these, but this gate ran first and rejected the line, so
    # "BA Fine Arts, 2023" was reported as no degree stated at all.
    r"|b\.?a\b|m\.?a\b|b\.?eng\b|m\.?eng\b|b\.?ed\b|llb\b|llm\b",
    re.I,
)

# Human-readable names for the levels above.
_LEVEL_LABELS = {
    "doctorate": "Doctorate",
    "master": "Master's",
    "bachelor": "Bachelor's",
    "associate": "Associate's",
    "diploma": "Diploma",
    "intermediate": "Intermediate",
}


def _level_label(level: str) -> str:
    return _LEVEL_LABELS.get(level, _title(level))


# "University of Management and Technology" is an institution, not a subject.
# Scanning it for field names credited the candidate with a management degree,
# so institution phrases are removed before the field scan runs.
_INSTITUTION_PHRASE = re.compile(
    r"\b(?:university|universiti|institute|college|school|academy|polytechnic)\b"
    r"(?:\s+of\s+[a-z&\s]+?)?(?=,|\||$)",
    re.I,
)


def _fields_in(fragment: str) -> list[str]:
    normalized = _INSTITUTION_PHRASE.sub(" ", _normalize(fragment))
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for code, field in _DEGREE_CODE_FIELDS.items():
        if re.search(rf"\b{code}\b", normalized):
            normalized = f"{normalized} {field}"

    found: list[str] = []
    for field in _ALL_FIELDS:
        if re.search(rf"\b{re.escape(field)}\b", normalized):
            if not any(field in existing for existing in found):
                found.append(field)
    return found


def _degree_level_in(fragment: str) -> str:
    normalized = _normalize(fragment)
    best_level = ""
    best_rank = 0
    for pattern, level in _DEGREE_PATTERNS:
        if re.search(pattern, normalized, re.I):
            rank = DEGREE_LEVELS[level]
            if rank > best_rank:
                best_rank = rank
                best_level = level
    return best_level


def extract_degrees(text: str) -> list[dict[str, Any]]:
    """Every degree the document claims, highest level first."""
    degrees: list[dict[str, Any]] = []
    for line in _lines(text):
        if not _EDUCATION_CONTEXT.search(line):
            continue
        level = _degree_level_in(line)
        if not level:
            continue
        degrees.append(
            {
                "level": level,
                "rank": DEGREE_LEVELS[level],
                "fields": _fields_in(line),
                "raw": line.strip()[:120],
            }
        )
    degrees.sort(key=lambda item: item["rank"], reverse=True)
    return degrees


def extract_education_requirement(job_text: str) -> dict[str, Any]:
    """The education bar the posting sets, if it sets one."""
    for line in _lines(job_text):
        if not _EDUCATION_CONTEXT.search(line):
            continue
        level = _degree_level_in(line)
        if not level:
            continue
        return {
            "level": level,
            "rank": DEGREE_LEVELS[level],
            "fields": _fields_in(line),
            "related_allowed": bool(_RELATED_FIELD_PHRASE.search(line)),
            "raw": line.strip()[:160],
        }
    return {"level": "", "rank": 0, "fields": [], "related_allowed": False, "raw": ""}


def _groups_of(fields: Iterable[str]) -> set[str]:
    return {_FIELD_GROUP_LOOKUP[field] for field in fields if field in _FIELD_GROUP_LOOKUP}


def check_education(degrees: list[dict[str, Any]], requirement: dict[str, Any]) -> dict[str, Any]:
    """Match a candidate's degrees against the posting's education requirement.

    Exact degree names are never required. "BS Data Science" satisfies
    "BS Computer Science or related field" because both sit in the computing
    group - the same judgement a human screener makes without thinking about it.
    """
    if not requirement.get("rank"):
        if degrees:
            top = degrees[0]
            return {
                "status": MATCH,
                "detail": f"The posting states no education requirement; the candidate holds a {_level_label(top['level'])} qualification.",
            }
        return {"status": UNKNOWN, "detail": "Neither document states an education requirement."}

    if not degrees:
        return {
            "status": UNKNOWN,
            "detail": f"The posting asks for a {_level_label(requirement['level'])} degree; the resume does not state one.",
        }

    required_rank = int(requirement["rank"])
    required_fields = list(requirement.get("fields") or [])
    required_groups = _groups_of(required_fields)

    best: dict[str, Any] | None = None
    best_quality = -1

    for degree in degrees:
        if degree["rank"] < required_rank:
            continue

        candidate_fields = degree["fields"]
        if not required_fields:
            quality = 2  # level is all that was asked for
        elif any(field in required_fields for field in candidate_fields):
            quality = 3  # named field matches outright
        elif required_groups & _groups_of(candidate_fields):
            quality = 2  # related field
        elif not candidate_fields:
            quality = 1  # right level, field unreadable
        else:
            quality = 0  # right level, unrelated field

        if quality > best_quality:
            best_quality = quality
            best = degree

    if best is None:
        top = degrees[0]
        return {
            "status": FAIL,
            "detail": (
                f"The posting asks for a {_level_label(requirement['level'])} degree; the resume shows "
                f"{_level_label(top['level'])} level."
            ),
        }

    field_label = ", ".join(_title(field) for field in best["fields"]) or "an unspecified field"
    required_label = ", ".join(_title(field) for field in required_fields) or "any field"

    if best_quality >= 3:
        return {"status": MATCH, "detail": f"{_level_label(best['level'])} in {field_label} matches the requirement."}
    if best_quality == 2:
        return {
            "status": MATCH,
            "detail": f"{_level_label(best['level'])} in {field_label} is a related field to {required_label}.",
        }
    if best_quality == 1:
        return {
            "status": MATCH,
            "detail": "Degree level meets the requirement; the subject is not stated in the resume.",
        }
    return {
        "status": BORDERLINE,
        "detail": (
            f"Degree level is met ({_level_label(best['level'])} in {field_label}), but the subject falls "
            f"outside {required_label}."
        ),
    }


# ---------------------------------------------------------------------------
# Work mode
# ---------------------------------------------------------------------------
_WORK_MODE_LABELS = {
    "remote-global": "Remote (worldwide)",
    "remote": "Remote",
    "hybrid": "Hybrid",
    "onsite": "On-site",
    "unknown": "Not stated",
}


def check_work_mode(work_mode: str, location_status: str) -> dict[str, Any]:
    label = _WORK_MODE_LABELS.get(work_mode, "Not stated")
    if work_mode in {"remote", "remote-global"}:
        return {"status": MATCH, "detail": f"{label} role, with no on-site attendance requirement."}
    if work_mode == "unknown":
        return {"status": UNKNOWN, "detail": "The posting does not state a work mode."}
    if location_status == MATCH:
        return {"status": MATCH, "detail": f"{label} role, and the candidate is already in the area."}
    if location_status == BORDERLINE:
        return {"status": BORDERLINE, "detail": f"{label} role; attendance depends on the candidate relocating."}
    if location_status == FAIL:
        return {"status": FAIL, "detail": f"{label} role, which the candidate cannot attend from their current location."}
    return {"status": UNKNOWN, "detail": f"{label} role; the candidate's location is not stated."}


# ---------------------------------------------------------------------------
# Screening entry point
# ---------------------------------------------------------------------------
_OVERALL_LABELS = {
    MATCH: "Eligible",
    BORDERLINE: "Needs Recruiter Review",
    UNKNOWN: "Needs Recruiter Review",
    FAIL: "Not Eligible",
}

_CHECK_LABELS = {
    MATCH: "Match",
    BORDERLINE: "Review",
    FAIL: "Not Met",
    UNKNOWN: "Not Stated",
}

# How each concern is named when it is raised. "Blocked on Location" describes
# the tool's internal state; "Location Mismatch" is what a recruiter would
# write on a shortlist. Purely presentational - the verdicts these names
# describe are decided elsewhere and are completely unchanged.
_ISSUE_NAMES = {
    ("location", FAIL): "Location Mismatch",
    ("location", BORDERLINE): "Location Requires Relocation",
    ("location", UNKNOWN): "Location Not Stated",
    ("experience", FAIL): "Experience Below Requirement",
    ("experience", BORDERLINE): "Experience Slightly Below Requirement",
    ("experience", UNKNOWN): "Experience Not Stated",
    ("education", FAIL): "Education Below Requirement",
    ("education", BORDERLINE): "Education Field Mismatch",
    ("education", UNKNOWN): "Education Not Stated",
    ("work_mode", FAIL): "Work Mode Not Feasible",
    ("work_mode", BORDERLINE): "Work Mode Requires Relocation",
    ("work_mode", UNKNOWN): "Work Mode Not Stated",
}


def issue_name(check: dict[str, Any]) -> str:
    """Recruiter-facing name for whatever this check is flagging."""
    key = check.get("key", "")
    # Deliberately the raw status. _effective_status softens a failing
    # experience check so it cannot block a hire - useful for the verdict,
    # wrong for the wording, which should still say how far short it is.
    status = check.get("status", UNKNOWN)
    if status == MATCH:
        return ""
    return _ISSUE_NAMES.get((key, status), str(check.get("title", "")).strip())

# Presentation hints resolved here rather than in the template, so the mapping
# from verdict to colour lives next to the verdicts themselves.
_STATUS_ICONS = {MATCH: "\u2705", BORDERLINE: "\u26a0", FAIL: "\u274c", UNKNOWN: "\u2753"}
_STATUS_PILLS = {
    MATCH: "score-good",
    BORDERLINE: "score-average",
    UNKNOWN: "score-average",
    FAIL: "score-poor",
}


# Checks that can genuinely block a hire. Experience is deliberately absent:
# being a year or two short is a conversation, not a rejection, so a shortfall
# routes the candidate to a human instead of closing the door. Location, work
# mode and education are the checks a recruiter actually cannot waive.
_BLOCKING_CHECKS = {"location", "work_mode", "education"}


def _effective_status(check: dict[str, Any]) -> str:
    """The status this check contributes to the overall verdict.

    A non-blocking check can never push the candidate to Not Eligible; the
    worst it can do is ask for a human decision.
    """
    status = check.get("status", UNKNOWN)
    if status == FAIL and check.get("key") not in _BLOCKING_CHECKS:
        return BORDERLINE
    return status


def _decorate(check: dict[str, Any]) -> dict[str, Any]:
    # Coloured by the status it actually contributes. A shortfall in years is
    # shown in amber with the "Below Requirement" wording rather than a red
    # cross, because it does not disqualify anyone on its own.
    status = _effective_status(check)
    check["icon"] = _STATUS_ICONS.get(status, _STATUS_ICONS[UNKNOWN])
    check["pill"] = _STATUS_PILLS.get(status, "score-average")
    # Recruiter-facing name for the concern, blank when there isn't one.
    check["issue"] = issue_name(check)
    return check


def screen_eligibility(
    *,
    resume_text: str,
    job_text: str,
    resume_experience_text: str = "",
    resume_education_text: str = "",
    job_education_text: str = "",
) -> dict[str, Any]:
    """Run every eligibility check and summarise the outcome.

    The result is presented next to the ATS score and never folded into it.
    """
    work_mode = extract_work_mode(job_text)

    candidate_place = extract_location(resume_text, header_lines=12)
    job_place = extract_location(job_text)

    location = check_location(candidate_place, job_place, work_mode)
    mode = check_work_mode(work_mode, location["status"])

    # One normalized analysis, consumed by every consumer below.
    experience_analysis = analyze_experience(job_text, resume_text, resume_experience_text)
    experience_requirement = experience_analysis["requirement"]
    required_years = experience_analysis["required_experience"]
    candidate_experience = {
        "years": experience_analysis["detected_experience"],
        "source": experience_analysis["detected_source"],
    }
    experience = experience_analysis["verdict"]

    degrees = extract_degrees(resume_education_text or resume_text)
    requirement = extract_education_requirement(job_education_text or job_text)
    education = check_education(degrees, requirement)

    experience_rows = _experience_rows_from(experience_analysis)

    checks = [
        {
            "key": "location",
            "title": "Location",
            "status": location["status"],
            "label": _CHECK_LABELS[location["status"]],
            "detail": location["detail"],
            "candidate_value": format_location(candidate_place),
            "job_value": format_location(job_place),
            "values": [
                {"label": "Candidate Location", "value": format_location(candidate_place)},
                {"label": "Job Location", "value": format_location(job_place)},
            ],
        },
        {
            "key": "experience",
            "title": "Experience",
            "status": experience["status"],
            "label": experience.get("label") or _CHECK_LABELS[experience["status"]],
            "detail": experience["detail"],
            "candidate_value": experience_analysis["detected_label"],
            "job_value": experience_analysis["required_label"],
            # Three labelled rows rather than two raw figures, so the gap the
            # decision turns on is stated instead of inferred by the reader.
            "values": experience_rows,
        },
        {
            "key": "education",
            "title": "Education",
            "status": education["status"],
            "label": _CHECK_LABELS[education["status"]],
            "detail": education["detail"],
            "candidate_value": (
                f"{_level_label(degrees[0]['level'])}"
                + (f" - {', '.join(_title(f) for f in degrees[0]['fields'])}" if degrees[0]["fields"] else "")
                if degrees
                else "Not stated"
            ),
            "job_value": (
                f"{_level_label(requirement['level'])}"
                + (f" - {', '.join(_title(f) for f in requirement['fields'])}" if requirement["fields"] else "")
                + (" or related" if requirement.get("related_allowed") else "")
                if requirement.get("rank")
                else "Not stated"
            ),
            "values": [
                {
                    "label": "Required Education",
                    "value": (
                        f"{_level_label(requirement['level'])}"
                        + (f" - {', '.join(_title(f) for f in requirement['fields'])}" if requirement["fields"] else "")
                        + (" or related" if requirement.get("related_allowed") else "")
                        if requirement.get("rank")
                        else "Not stated"
                    ),
                },
                {
                    "label": "Candidate Education",
                    "value": (
                        f"{_level_label(degrees[0]['level'])}"
                        + (f" - {', '.join(_title(f) for f in degrees[0]['fields'])}" if degrees[0]["fields"] else "")
                        if degrees
                        else "Not stated"
                    ),
                },
            ],
        },
        {
            "key": "work_mode",
            "title": "Work Mode",
            "status": mode["status"],
            "label": _CHECK_LABELS[mode["status"]],
            "detail": mode["detail"],
            # Previously this repeated the candidate's city, which read oddly
            # against a work mode ("Lahore, Pakistan against On-site").
            "candidate_value": (
                f"Based in {format_location(candidate_place)}"
                if format_location(candidate_place) != "Not stated"
                else "Not stated"
            ),
            "job_value": _WORK_MODE_LABELS.get(work_mode, "Not stated"),
            "values": [
                {"label": "Job Work Mode", "value": _WORK_MODE_LABELS.get(work_mode, "Not stated")},
                {
                    "label": "Candidate Base",
                    "value": (
                        format_location(candidate_place)
                        if format_location(candidate_place) != "Not stated"
                        else "Not stated"
                    ),
                },
            ],
        },
    ]

    checks = [_decorate(check) for check in checks]

    worst = max(
        (_effective_status(check) for check in checks),
        key=lambda status: _STATUS_RANK[status],
    )
    blocking = [
        issue_name(check)
        for check in checks
        if check["status"] == FAIL and check["key"] in _BLOCKING_CHECKS
    ]
    review = [
        issue_name(check)
        for check in checks
        if _effective_status(check) in {BORDERLINE, UNKNOWN}
    ]

    if worst == FAIL:
        summary = f"{'; '.join(blocking)}. Recommend recruiter review before rejecting."
    elif review:
        verb = "requires" if len(review) == 1 else "require"
        if all(_effective_status(check) == UNKNOWN for check in checks):
            summary = (
                "Eligibility could not be verified: the documents state none of the "
                "location, work mode, experience or education criteria."
            )
        else:
            summary = (
                f"Meets the mandatory criteria. {'; '.join(review)} {verb} recruiter review."
            )
    else:
        summary = "Meets all stated eligibility requirements."

    return {
        "checks": checks,
        "overall_status": worst,
        "overall_label": _OVERALL_LABELS[worst],
        "overall_icon": _STATUS_ICONS.get(worst, _STATUS_ICONS[UNKNOWN]),
        "overall_pill": _STATUS_PILLS.get(worst, "score-average"),
        "summary": summary,
        "blocking": blocking,
        "review": review,
        "work_mode": work_mode,
        "work_mode_label": _WORK_MODE_LABELS.get(work_mode, "Not stated"),
        "candidate_location": format_location(candidate_place),
        "job_location": format_location(job_place),
        "candidate_years": candidate_experience["years"],
        "candidate_years_source": candidate_experience["source"],
        "required_years": required_years,
        "experience_requirement": experience_requirement,
        "experience_analysis": experience_analysis,
        "experience_rows": experience_rows,
        "candidate_degrees": degrees[:3],
        "education_requirement": requirement,
        "education_status": education["status"],
        # The scorer reuses this rather than re-deriving the same judgement,
        # so the Education section score and the Education eligibility row can
        # never disagree with each other.
        "education_check": education,
    }