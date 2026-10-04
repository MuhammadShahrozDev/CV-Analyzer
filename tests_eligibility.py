"""Regression suite for eligibility screening, skill tiering and input parity.

Complements tests_calibration.py, which checks that scores rank resumes
sensibly. This file checks the behaviours added on top of that:

  * required vs preferred vs implied skills are kept apart
  * a related degree satisfies an "or related field" requirement
  * location / work mode / experience screening produces the right verdicts
  * the same posting scores identically whether pasted, uploaded or linked

Run with:  python tests_eligibility.py
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from utils.eligibility import screen_eligibility
from utils.extractor import extract_skills_from_text, load_skills_config
from utils.matcher import build_match_report
from utils.parser import normalize_job_description
from utils.preprocessing import preprocess_text
from utils.scorer import build_ats_report

cfg = load_skills_config(BASE / "skills.json")

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "OK  " if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def analyze(resume_text: str, job_raw: str) -> dict:
    """Run the full pipeline exactly as app.py does."""
    job_flat, job_layout = normalize_job_description(job_raw)
    resume_flat = " ".join(resume_text.split())

    report = build_match_report(
        resume_text=resume_flat,
        job_text=job_flat,
        resume_clean=preprocess_text(resume_text),
        job_clean=preprocess_text(job_layout),
        resume_skills=extract_skills_from_text(resume_text, cfg),
        job_skills=extract_skills_from_text(job_layout, cfg),
        skills_config=cfg,
        resume_layout_text=resume_text,
        job_layout_text=job_layout,
    )
    return build_ats_report(report)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
JD = """Data Analyst
Location: Lahore, Pakistan
On-site position

Responsibilities:
- Develop dashboards for business stakeholders
- Analyse large datasets and produce actionable insights
- Write and optimise SQL queries

Requirements:
- Bachelor's degree in Computer Science or related field
- 2+ years of experience in data analysis
- Proficiency in Python and SQL
- Experience with Power BI

Preferred Qualifications:
- Experience with Tableau
- Knowledge of AWS and Snowflake

Benefits:
- Health insurance, annual bonus, 25 days paid leave
"""

RESUME = """Sabahat Ali
Data Analyst
Lahore, Pakistan | sabahat@example.com

PROFESSIONAL SUMMARY
Data analyst with 3 years of experience turning operational data into reporting.

TECHNICAL SKILLS
Python, SQL, Power BI, Pandas, Excel, Git

EDUCATION
BS Data Science, University of Management and Technology, 2023

WORK EXPERIENCE
Data Analyst, Systems Ltd (2023 - Present)
- Developed interactive Power BI dashboards for business stakeholders
- Analysed large datasets and produced actionable insights
- Wrote and optimised complex SQL queries, cutting runtime from 40s to 3s
"""


# ---------------------------------------------------------------------------
print("\n1. Skill tiers come only from the job description")
report = analyze(RESUME, JD)

check(
    "Tableau/AWS/Snowflake land in the preferred tier",
    {"Tableau", "AWS", "Snowflake"} <= set(report["preferred_skills"]),
    str(report["preferred_skills"]),
)
check(
    "Preferred-only skills never appear as missing required",
    not ({"Tableau", "AWS", "Snowflake"} & set(report["missing_skills"])),
    str(report["missing_skills"]),
)
check(
    "Python/SQL/Power BI are treated as required",
    {"Python", "SQL", "Power BI"} <= set(report["required_skills"]),
    str(report["required_skills"]),
)
check(
    "Nothing outside the job description is reported as missing",
    set(report["missing_skills"]) <= set(report["required_skills"]),
    str(report["missing_skills"]),
)
check(
    "A resume holding every must-have has no required gaps",
    report["missing_skills"] == [],
    str(report["missing_skills"]),
)

# Same posting with the preferred block promoted to a hard requirement.
JD_HARD = JD.replace("Preferred Qualifications:", "Additional Requirements:")
hard = analyze(RESUME, JD_HARD)
check(
    "The same skills DO count as gaps when the posting requires them",
    bool({"Tableau", "AWS", "Snowflake"} & set(hard["missing_skills"])),
    str(hard["missing_skills"]),
)
check(
    "Preferred wording scores higher than required wording",
    report["ats_score"] > hard["ats_score"],
    f"{report['ats_score']} vs {hard['ats_score']}",
)


# ---------------------------------------------------------------------------
print("\n2. Education accepts related degrees")
check(
    "BS Data Science satisfies 'Computer Science or related field'",
    report["eligibility"]["education_status"] == "match",
    str(report["education_assessment"]),
)
check(
    "Education section scores well when the degree qualifies",
    report["education_score"] >= 85,
    str(report["education_score"]),
)

for degree in ("BS Computer Science", "BS Software Engineering", "BSIT",
               "BS Artificial Intelligence", "MS Statistics"):
    swapped = RESUME.replace("BS Data Science", degree)
    result = screen_eligibility(resume_text=swapped, job_text=JD)
    check(f"{degree} satisfies the requirement", result["education_status"] == "match")

unrelated = RESUME.replace("BS Data Science", "BA Fine Arts")
result = screen_eligibility(resume_text=unrelated, job_text=JD)
check(
    "An unrelated degree is flagged rather than silently passed",
    result["education_status"] != "match",
    result["education_status"],
)


# ---------------------------------------------------------------------------
print("\n3. Location, work mode and experience screening")

CASES = [
    ("same city, on-site", "Lahore, Pakistan", "Location: Lahore\nOn-site position", "match"),
    ("remote worldwide", "Lahore, Pakistan", "Fully remote, work from anywhere.", "match"),
    ("foreign on-site", "Lahore, Pakistan", "Location: San Francisco, United States\nOn-site position", "fail"),
    ("same country, hybrid", "Karachi, Pakistan", "Job Location: Islamabad\nHybrid - 3 days in office", "borderline"),
    ("no location stated", "Lahore, Pakistan", "We are hiring an analyst.", "unknown"),
]
for label, candidate_line, job_line, expected in CASES:
    resume = f"Test Candidate\n{candidate_line}\nBS Data Science\n3 years of experience"
    result = screen_eligibility(resume_text=resume, job_text=job_line)
    location = next(c for c in result["checks"] if c["key"] == "location")
    check(f"location: {label} -> {expected}", location["status"] == expected, location["status"])

EXPERIENCE_CASES = [
    ("meets the bar", "5 years of experience", "5+ years of experience required", "match"),
    ("one year short", "2 years of experience", "3+ years of experience required", "borderline"),
    ("well short", "1 year of experience", "8+ years of experience required", "fail"),
    ("no minimum stated", "2 years of experience", "We want a great analyst.", "match"),
]
for label, resume_line, job_line, expected in EXPERIENCE_CASES:
    resume = f"Test Candidate\nLahore, Pakistan\n{resume_line}"
    result = screen_eligibility(resume_text=resume, job_text=job_line)
    experience = next(c for c in result["checks"] if c["key"] == "experience")
    check(f"experience: {label} -> {expected}", experience["status"] == expected, experience["status"])

short = screen_eligibility(
    resume_text="Test Candidate\nLahore, Pakistan\n1 year of experience\nBS Data Science",
    job_text="Location: Lahore\n8+ years of experience required\nBachelor in Computer Science or related field",
)
check(
    "A shortfall never auto-rejects: it routes to a human",
    short["overall_label"] in {"Needs Recruiter Review", "Not Eligible"}
    and "Experience" not in short.get("blocking", []),
    f"{short['overall_label']} / blocking={short.get('blocking')}",
)


# ---------------------------------------------------------------------------
print("\n4. The three job-description inputs agree")

# What a URL fetch looks like before cleaning: the posting wrapped in site
# chrome. What a PDF or DOCX looks like: the same posting with ragged spacing.
FROM_URL = (
    "Skip to main content\nSign in\nWe use cookies\nAccept all\nApply now\n\n"
    + JD
    + "\nApply now\nSave job\nReport this job\nSimilar jobs\nPrivacy Policy\n(c) 2026 JobSite\n"
)
FROM_FILE = "\n\n".join(line + "   " for line in JD.split("\n"))

pasted = analyze(RESUME, JD)
from_url = analyze(RESUME, FROM_URL)
from_file = analyze(RESUME, FROM_FILE)

check(
    "pasted vs URL produce the same ATS score",
    pasted["ats_score"] == from_url["ats_score"],
    f"{pasted['ats_score']} vs {from_url['ats_score']}",
)
check(
    "pasted vs uploaded file produce the same ATS score",
    pasted["ats_score"] == from_file["ats_score"],
    f"{pasted['ats_score']} vs {from_file['ats_score']}",
)
check(
    "the three inputs agree on missing required skills",
    pasted["missing_skills"] == from_url["missing_skills"] == from_file["missing_skills"],
)
check(
    "site chrome never becomes a missing keyword",
    not ({"cookies", "privacy", "apply"} & set(from_url["missing_keywords"])),
    str(from_url["missing_keywords"][:10]),
)


# ---------------------------------------------------------------------------
print("\n5. Eligibility stays out of the ATS score")

INELIGIBLE_JD = JD.replace("Location: Lahore, Pakistan", "Location: San Francisco, United States")
ineligible = analyze(RESUME, INELIGIBLE_JD)
check(
    "moving the job abroad does not change the ATS score",
    abs(ineligible["ats_score"] - pasted["ats_score"]) < 0.01,
    f"{pasted['ats_score']} vs {ineligible['ats_score']}",
)
check(
    "but it does change the eligibility verdict",
    ineligible["eligibility"]["overall_label"] == "Not Eligible",
    ineligible["eligibility"]["overall_label"],
)
check(
    "every eligibility row carries a verdict and an explanation",
    all(c.get("status") and c.get("detail") for c in ineligible["eligibility"]["checks"]),
)


# ---------------------------------------------------------------------------
print("\n6. Recommendations follow the stated priority order")
gap_resume = RESUME.replace("Python, SQL, Power BI, Pandas, Excel, Git", "Excel, Word")
gaps = analyze(gap_resume, JD)
recommendations = gaps["recommendations"]
required_gaps = set(gaps["missing_skills"])

first_required = next(
    (i for i, text in enumerate(recommendations) if any(skill in text for skill in required_gaps)),
    None,
)
first_optional = next((i for i, text in enumerate(recommendations) if text.startswith("Optional:")), None)
check("a missing required skill is raised first", first_required == 0, str(recommendations[:1]))
check(
    "optional extras are ranked below everything else",
    first_optional is None or (first_required is not None and first_optional > first_required),
    f"required@{first_required} optional@{first_optional}",
)
check("recommendations stay to a readable length", len(recommendations) <= 8, str(len(recommendations)))


# ---------------------------------------------------------------------------
print("\n7. Edge cases do not crash")
for label, resume, job in [
    ("empty job description", RESUME, "   "),
    ("empty resume", "   ", JD),
    ("no headings anywhere", "Python SQL Power BI dashboards", "Need Python and SQL for dashboards"),
    ("preferred block only", RESUME, "Analyst\nNice to have:\n- Tableau\n- AWS"),
]:
    try:
        result = analyze(resume, job)
        check(f"{label} -> ATS {result['ats_score']}", True)
    except Exception as exc:  # pragma: no cover - the point is that it should not happen
        check(f"{label} raised {type(exc).__name__}", False, str(exc))


# ---------------------------------------------------------------------------
# Experience requirement extraction.
#
# This section exists because the extractor was rewritten several times and
# each rewrite broke a case the previous one handled. Every phrasing below has
# failed at some point, so each line is a regression that actually happened
# rather than a hypothetical. Add to it whenever a posting is misread.
# ---------------------------------------------------------------------------
print("\n8. Experience requirements are read from the wording")

from utils.eligibility import (  # noqa: E402 - grouped with its own section
    analyze_experience,
    extract_experience_requirement,
    format_required_experience,
)

REQUIRED_PHRASINGS = [
    # (job description text, expected minimum, expected top of range)
    ("4 or more years of experience building production software", 4, None),
    ("4+ years of experience", 4, None),
    ("4 years of experience", 4, None),
    ("at least 4 years of experience", 4, None),
    ("minimum 4 years of experience", 4, None),
    ("a minimum of 4 years of experience", 4, None),
    ("experience of 4 years or more", 4, None),
    ("3-5 years of experience", 3, 5),
    ("3 to 5 years of experience", 3, 5),
    ("2-4 years of experience", 2, 4),
    ("2 to 4 years of relevant experience", 2, 4),
    ("5+ years building production software", 5, None),
    ("5 years building production software", 5, None),
    ("4 years building production systems", 4, None),
    ("3 years of professional experience", 3, None),
    ("2 years of industry experience", 2, None),
    ("1 year of relevant experience", 1, None),
    ("2+ years in software engineering", 2, None),
    ("3+ years of frontend development experience", 3, None),
    ("2+ years of Python experience", 2, None),
    ("5+ years of experience with React and TypeScript", 5, None),
    # Written-out numbers.
    ("four or more years of experience", 4, None),
    ("at least four years of experience", 4, None),
    ("five years of experience", 5, None),
    ("three years of industry experience", 3, None),
    ("two or more years", 2, None),
]

for text, expected_min, expected_max in REQUIRED_PHRASINGS:
    found = extract_experience_requirement(text)
    check(
        f'"{text[:46]}" -> {expected_min}',
        found["min_years"] == float(expected_min) and found["max_years"] == (
            float(expected_max) if expected_max is not None else None
        ),
        f"got min={found['min_years']} max={found['max_years']}",
    )

# "N or more" must never be read as N+1. This was a real defect: unanchored
# number matching let a figure be taken from inside a longer token.
for value in range(1, 21):
    for phrasing in (
        f"{value} or more years of experience",
        f"{value}+ years of experience",
        f"at least {value} years of experience",
    ):
        check(
            f"{phrasing} -> exactly {value}",
            extract_experience_requirement(phrasing)["min_years"] == float(value),
        )

print("\n9. Numbers that are not experience requirements")

NOT_REQUIREMENTS = [
    "Backed by a16z and Y Combinator, with $75M raised in Series A funding",
    "Already serving more than 20% of the auto lending industry",
    "Processing millions of real customer calls every day",
    "Cash flow positive with mid eight figure ARR achieved in under two years",
    "Our team typically works around 60 hours per week",
    "Beginning at 8:00 AM, with four days spent collaborating in person",
    "Founded 2 years ago in San Francisco",
    "Over the past 3 years we have grown quickly",
    "Contract length: 2 years",
    "Visa sponsorship valid for 3 years",
    "Over 125 years of combined team experience",
    "Backed by founders with more than 10 years of experience",
    "A 15 year old codebase",
    "Established in 2015",
    "No prior experience required",
    "No minimum experience required.",
    "We are looking for a great engineer who cares about users",
]

for text in NOT_REQUIREMENTS:
    found = extract_experience_requirement(text)
    check(
        f'no requirement from "{text[:44]}"',
        found["min_years"] is None,
        f"got {found['min_years']} from {found['raw']!r}",
    )

print("\n10. General requirements outrank tool-specific ones")

both = extract_experience_requirement(
    "2+ years of React experience and 5+ years of software engineering experience"
)
check(
    "overall bar is the general requirement, not the tool-specific one",
    both["min_years"] == 5.0,
    f"got {both['min_years']}",
)
check(
    "the tool-specific figure is preserved separately",
    any(item["years"] == 2.0 for item in both.get("skill_requirements", [])),
    str(both.get("skill_requirements")),
)

# A requirement sharing a line with unrelated prose must survive. Judging the
# whole line as one unit once deleted the requirement and left "Not stated".
for tail in (
    "and you ship every week",
    "maintaining contract integrations",
    "since our last platform migration",
    "owning the next generation of tooling",
):
    text = f"What you'll bring:\n- 4 or more years of experience building software, {tail}\n"
    check(
        f"requirement survives a clause about: {tail[:34]}",
        extract_experience_requirement(text)["min_years"] == 4.0,
    )

print("\n11. The normalized experience analysis is self-consistent")

ANALYSIS_CASES = [
    ("4 or more years of experience", "Intern (Jul 2026 - Present)", "below_requirement"),
    ("2+ years of experience", "5 years of experience", "match"),
    ("8+ years of experience", "2 years of experience", "below_requirement"),
    ("We want a great engineer", "3 years of experience", "not_stated"),
]

for job_text, resume_line, expected_status in ANALYSIS_CASES:
    analysis = analyze_experience(job_text, f"Ali\nLahore, Pakistan\n{resume_line}\n")
    check(
        f"status for '{job_text[:28]}' is {expected_status}",
        analysis["status"] == expected_status,
        f"got {analysis['status']}",
    )
    required = analysis["required_experience"]
    detected = analysis["detected_experience"]
    if required is not None and detected is not None:
        shortfall = round(max(0.0, required - detected), 1)
        check(
            f"gap is arithmetic for '{job_text[:28]}'",
            (analysis["experience_gap"] or 0.0) == shortfall,
            f"gap={analysis['experience_gap']} expected {shortfall}",
        )
    check(
        f"no negative gap shown for '{job_text[:28]}'",
        "-" not in analysis["gap_label"],
        analysis["gap_label"],
    )

# One source of truth: the card, the label and the raw number must agree.
JOB_WITH_BAR = "What you'll bring\n4 or more years of experience building production software\n"
RESUME_JUNIOR = "Ali Raza\nLahore, Pakistan\nWORK EXPERIENCE\nIntern (Jul 2026 - Present)\n"
screened = screen_eligibility(resume_text=RESUME_JUNIOR, job_text=JOB_WITH_BAR)
experience_check = next(c for c in screened["checks"] if c["key"] == "experience")
row_value = experience_check["values"][0]["value"]
check(
    "card row, card job_value and the analysis label all agree",
    row_value == experience_check["job_value"] == screened["experience_analysis"]["required_label"],
    f"{row_value!r} / {experience_check['job_value']!r}",
)
check(
    "an extracted requirement is never displayed as 'Not stated'",
    row_value != "Not stated" and screened["required_years"] == 4.0,
    f"row={row_value!r} required_years={screened['required_years']}",
)
check(
    "range requirements display as a range",
    format_required_experience(extract_experience_requirement("3-5 years of experience"))
    == "3-5 years",
)


print("\n" + "=" * 70)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for item in FAILURES:
        print(f"  - {item}")
    sys.exit(1)
print("All eligibility, tiering and parity checks passed.")