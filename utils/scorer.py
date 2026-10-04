from __future__ import annotations

from typing import Any

# Rebalanced so the sections a recruiter actually reads carry the weight.
# Education moved up from a token 0.05: it is a stated hiring requirement in
# most postings, and the analyzer now judges it properly (see the education
# scoring in matcher.py) rather than on wording similarity. professional_summary
# moved down, because it is the most wording-driven section of all and was
# lending similarity a second, hidden vote.
SECTION_WEIGHTS = {
    "technical_skills": 0.38,
    "work_experience": 0.26,
    "projects": 0.13,
    "professional_summary": 0.07,
    "education": 0.09,
    "soft_skills": 0.07,
}

SECTION_DISPLAY_NAMES = {
    "technical_skills": "Technical Skills",
    "work_experience": "Experience",
    "projects": "Projects",
    "professional_summary": "Summary",
    "education": "Education",
    "soft_skills": "Soft Skills",
}

# How the final score is composed. Section roll-up alone was too easy to
# distort: one absent section could drag a strong resume down. Direct evidence
# (which requirements are met, which required skills are present) is what a
# recruiter actually reads, so it carries the majority.
COMPOSITE_WEIGHTS = {
    "skill": 0.44,
    "requirement": 0.34,
    "keyword": 0.14,
    "similarity": 0.08,
}

CORE_VS_SECTIONS = (0.65, 0.35)

# Direct evidence, expressed on the same 0-100 scale as the score itself.
# Required-skill coverage and requirement coverage are what a recruiter checks
# first, so when both are strong the score is not allowed to collapse just
# because the candidate described the same work in different words.
#
# The floor is deliberately conservative: it only engages above a genuinely
# strong alignment, it recovers a fraction of that alignment rather than all of
# it, and it is applied *before* the substantiation dampener - so a resume that
# lists skills without evidencing them is still marked down afterwards.
ALIGNMENT_FLOOR_AT = 70.0
ALIGNMENT_FLOOR_SHARE = 0.86

# A resume needs most of its claimed skills evidenced in experience or projects
# to score at full strength. At zero substantiation it retains only the floor.
SUBSTANTIATION_FLOOR = 0.55
SUBSTANTIATION_TARGET = 0.60

# Raw composite is not a recruiter-facing percentage. These anchors map it onto
# the range hiring tools report, without lifting the floor: an unrelated resume
# still lands in single digits. Verified against both a matched and a
# deliberately mismatched resume.
_SCORE_ANCHORS = (
    (0.0, 0.0),
    (10.0, 8.0),
    (25.0, 24.0),
    (40.0, 45.0),
    (55.0, 65.0),
    (70.0, 79.0),
    (85.0, 91.0),
    (100.0, 100.0),
)


def _interpolate(value: float, anchors: tuple[tuple[float, float], ...]) -> float:
    value = max(0.0, min(100.0, value))
    for index in range(len(anchors) - 1):
        low_x, low_y = anchors[index]
        high_x, high_y = anchors[index + 1]
        if low_x <= value <= high_x:
            if high_x == low_x:
                return low_y
            ratio = (value - low_x) / (high_x - low_x)
            return low_y + ratio * (high_y - low_y)
    return anchors[-1][1]


def calibrate_score(raw: float) -> float:
    return round(_interpolate(float(raw), _SCORE_ANCHORS), 2)


def _weighted_score(keyword_score: float, skill_score: float, similarity_score: float) -> float:
    weighted = (0.35 * keyword_score) + (0.35 * skill_score) + (0.30 * similarity_score)
    return round(max(0.0, min(100.0, weighted)), 2)


def _score_label(score: float) -> str:
    if score >= 75:
        return "good"
    if score >= 50:
        return "average"
    return "poor"


def _section_scores(report: dict[str, Any]) -> dict[str, float]:
    section_analysis = report.get("section_analysis", {}) or {}
    scores: dict[str, float] = {}
    for section_name in SECTION_WEIGHTS:
        scores[section_name] = round(float(section_analysis.get(section_name, {}).get("score", 0.0)), 2)
    return scores


def _applicable_sections(report: dict[str, Any]) -> set[str]:
    """Sections that should count toward the roll-up.

    Two failure modes are avoided. A section the resume lacks and the posting
    never asks about previously contributed a hard zero, penalising a resume
    for omitting something nobody wanted. And a section the resume *has* but
    the posting never mentions (Education is the common case) was scored
    against an empty job section, producing a zero for perfectly good content.

    Core sections count whenever the resume has them. Everything else counts
    only when the posting actually asks. Remaining weights are renormalised.
    """
    section_analysis = report.get("section_analysis", {}) or {}
    job_sections = report.get("job_sections", {}) or {}

    # Education is a core section. It was previously counted only when the
    # posting carried its own Education heading - which postings rarely do -
    # so a candidate whose degree fully satisfied the requirement had that
    # result silently dropped from the roll-up.
    core_sections = {
        "technical_skills",
        "work_experience",
        "projects",
        "professional_summary",
        "education",
    }

    applicable: set[str] = set()
    for section_name in SECTION_WEIGHTS:
        detail = section_analysis.get(section_name, {}) or {}
        resume_has = bool(detail.get("section_present"))
        job_asks = bool((job_sections.get(section_name) or "").strip())

        if job_asks:
            applicable.add(section_name)
        elif resume_has and section_name in core_sections:
            applicable.add(section_name)

    # Never drop everything.
    return applicable or set(SECTION_WEIGHTS)


def _overall_section_score(section_scores: dict[str, float], applicable: set[str]) -> float:
    total_weight = sum(weight for name, weight in SECTION_WEIGHTS.items() if name in applicable)
    if total_weight <= 0:
        return 0.0
    weighted_total = sum(
        section_scores.get(name, 0.0) * weight
        for name, weight in SECTION_WEIGHTS.items()
        if name in applicable
    )
    return round(max(0.0, min(100.0, weighted_total / total_weight)), 2)


def _composite_score(report: dict[str, Any], section_scores: dict[str, float]) -> tuple[float, dict[str, float]]:
    keyword_score = float(report.get("keyword_match", {}).get("score", 0.0))
    skill_score = float(report.get("skill_match", {}).get("score", 0.0))
    similarity_score = float(report.get("similarity", 0.0))
    requirement_report = report.get("requirement_report", {}) or {}
    requirement_score = float(requirement_report.get("score", 0.0))

    # With no parseable requirements, redistribute that weight onto skills
    # rather than scoring the resume against nothing.
    weights = dict(COMPOSITE_WEIGHTS)
    if not requirement_report.get("evaluated"):
        weights["skill"] += weights.pop("requirement")
        requirement_score = 0.0

    core = (
        weights.get("skill", 0.0) * skill_score
        + weights.get("requirement", 0.0) * requirement_score
        + weights.get("keyword", 0.0) * keyword_score
        + weights.get("similarity", 0.0) * similarity_score
    )

    applicable = _applicable_sections(report)
    section_roll = _overall_section_score(section_scores, applicable)

    core_weight, section_weight = CORE_VS_SECTIONS
    raw = (core_weight * core) + (section_weight * section_roll)

    # Wording differences must not sink a resume that demonstrably does the job.
    alignment = (0.6 * skill_score) + (0.4 * requirement_score)
    floor = alignment * ALIGNMENT_FLOOR_SHARE if alignment >= ALIGNMENT_FLOOR_AT else 0.0
    floor_applied = floor > raw
    if floor_applied:
        raw = floor

    # Skills listed but never evidenced in experience or projects are claims,
    # not proof. Without this a resume of bare nouns and one filler bullet
    # outscored genuine experience. The dampener is bounded so a strong resume
    # with a couple of unproven extras is barely affected, while a stuffed one
    # loses a lot.
    substantiation = float(report.get("substantiation", 1.0) or 0.0)
    dampener = SUBSTANTIATION_FLOOR + (1.0 - SUBSTANTIATION_FLOOR) * min(1.0, substantiation / SUBSTANTIATION_TARGET)
    raw *= dampener

    components = {
        "alignment": round(alignment, 2),
        "alignment_floor_applied": floor_applied,
        "skill": round(skill_score, 2),
        "requirement": round(requirement_score, 2),
        "keyword": round(keyword_score, 2),
        "similarity": round(similarity_score, 2),
        "section_rollup": section_roll,
        "substantiation": round(substantiation * 100, 2),
        "substantiation_dampener": round(dampener, 4),
        "raw_composite": round(raw, 2),
    }
    return raw, components


def _best_and_weakest_sections(
    section_scores: dict[str, float],
    applicable: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    # Only sections that actually counted toward the score are described.
    # Naming Education as a weakness when the posting never asked about it,
    # and when it was excluded from the roll-up, is misleading advice.
    scoped = {
        name: score
        for name, score in section_scores.items()
        if applicable is None or name in applicable
    } or section_scores

    strongest = sorted(scoped.items(), key=lambda item: item[1], reverse=True)[:3]
    weakest = sorted(scoped.items(), key=lambda item: item[1])[:3]
    strong_labels = [SECTION_DISPLAY_NAMES[name] for name, score in strongest if score >= 70]
    weak_labels = [SECTION_DISPLAY_NAMES[name] for name, score in weakest if score < 70]
    return strong_labels, weak_labels


def _join_phrases(items: list[str], limit: int = 4) -> str:
    values = [item for item in items if item]
    if not values:
        return ""
    values = values[:limit]
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


# ---------------------------------------------------------------------------
# Education verdict (Task 5)
#
# A percentage on Education was never meaningful: a degree either satisfies the
# stated requirement or it does not. The underlying number still feeds the
# section roll-up, but what a recruiter reads is now a verdict and a reason.
# ---------------------------------------------------------------------------
EDUCATION_VERDICTS = {
    "match": {"label": "Match", "icon": "\u2713", "pill": "score-good"},
    "borderline": {"label": "Partial Match", "icon": "\u26a0", "pill": "score-average"},
    "fail": {"label": "Not Match", "icon": "\u2717", "pill": "score-poor"},
    "unknown": {"label": "Not Stated", "icon": "\u2013", "pill": "score-average"},
}


def build_education_verdict(report: dict[str, Any]) -> dict[str, str]:
    """Pass/fail verdict for Education, with the reason behind it."""
    assessment = report.get("education_assessment", {}) or {}
    status = assessment.get("status", "unknown")
    verdict = dict(EDUCATION_VERDICTS.get(status, EDUCATION_VERDICTS["unknown"]))
    verdict["status"] = status
    verdict["detail"] = assessment.get("detail", "") or "No education requirement could be read."
    return verdict


# ---------------------------------------------------------------------------
# Recruiter decision (Task 7)
#
# Deliberately not a function of the ATS score. A candidate can score 85% and
# still be unable to take the job, and a 60% resume in the right city with the
# right degree is worth a call. The decision is driven by the five things a
# recruiter actually screens on: education, experience, location, work mode
# and required technical skills.
# ---------------------------------------------------------------------------
DECISION_PROCEED = "Proceed to Interview"
DECISION_CAUTION = "Proceed with Caution"
DECISION_HOLD = "Hold"
DECISION_REJECT = "Reject"

_DECISION_PILLS = {
    DECISION_PROCEED: "score-good",
    DECISION_CAUTION: "score-average",
    DECISION_HOLD: "score-average",
    DECISION_REJECT: "score-poor",
}

# Technical coverage thresholds, expressed as the share of required technical
# skills the resume evidences.
_TECHNICAL_STRONG = 0.80
_TECHNICAL_WEAK = 0.50

# Checks a recruiter cannot waive. Experience is excluded on purpose: being
# short of a posted figure is a conversation, not a disqualification.
_HARD_CHECKS = ("location", "work_mode", "education")


def build_recruiter_decision(report: dict[str, Any]) -> dict[str, Any]:
    """Shortlisting decision, with the reasons that produced it."""
    eligibility = report.get("eligibility", {}) or {}
    checks = {check.get("key"): check for check in (eligibility.get("checks", []) or [])}

    required_technical = report.get("required_technical_skills", []) or []
    missing_technical = report.get("missing_required_technical_skills", []) or []
    coverage = (
        1.0
        if not required_technical
        else 1.0 - (len(missing_technical) / len(required_technical))
    )

    hard_failures = [
        checks[key]
        for key in _HARD_CHECKS
        if key in checks and checks[key].get("status") == "fail"
    ]
    concerns = [
        check
        for key, check in checks.items()
        if check.get("status") == "borderline"
        or (key == "experience" and check.get("status") == "fail")
    ]
    unknowns = [check for check in checks.values() if check.get("status") == "unknown"]

    reasons: list[str] = []
    for check in hard_failures:
        reasons.append(check.get("issue") or f"{check.get('title')} not met")
    for check in concerns:
        reasons.append(check.get("issue") or f"{check.get('title')} needs review")

    if missing_technical:
        reasons.append(
            f"{len(missing_technical)} of {len(required_technical)} required technical "
            f"skills not evidenced ({_join_phrases(sorted(missing_technical), limit=3)})"
        )

    if hard_failures and coverage < _TECHNICAL_WEAK:
        decision = DECISION_REJECT
        rationale = (
            "Fails a mandatory screening criterion and evidences under half of the "
            "required technical skills."
        )
    elif hard_failures:
        decision = DECISION_HOLD
        rationale = (
            "Technically capable, but a mandatory screening criterion is not met. "
            "Confirm before progressing or rejecting."
        )
    elif coverage < _TECHNICAL_WEAK:
        decision = DECISION_HOLD
        rationale = "Clears the screening criteria, but the technical gap is substantial."
    elif concerns or unknowns or coverage < _TECHNICAL_STRONG:
        decision = DECISION_CAUTION
        rationale = "Broadly suitable, with points to probe at screening stage."
    else:
        decision = DECISION_PROCEED
        rationale = (
            "Meets every screening criterion and evidences the required technical skills."
        )

    if not reasons:
        reasons.append("No screening concerns identified.")
    if unknowns and decision != DECISION_PROCEED:
        reasons.append(
            f"{_join_phrases([check.get('title', '') for check in unknowns], limit=3)} not stated"
        )

    return {
        "decision": decision,
        "pill": _DECISION_PILLS.get(decision, "score-average"),
        "rationale": rationale,
        "reasons": reasons[:5],
        "technical_coverage": round(coverage * 100, 1),
        "required_technical_count": len(required_technical),
        "missing_technical_count": len(missing_technical),
    }


def _skill_feedback(report: dict[str, Any]) -> dict[str, list[str]]:
    skill_match = report.get("skill_match", {}) or {}
    return {
        "matched": skill_match.get("matched", []) or [],
        "partial": skill_match.get("partial", []) or [],
        "missing_required": skill_match.get("missing_required", skill_match.get("missing", [])) or [],
        "missing_required_technical": skill_match.get("missing_required_technical", []) or [],
        "missing_required_soft": skill_match.get("missing_required_soft", []) or [],
        "missing_preferred": skill_match.get("missing_preferred", []) or [],
        "missing_inferred": skill_match.get("missing_inferred", []) or [],
        "preferred": skill_match.get("preferred_skills", []) or [],
    }


# Recruiter feedback is a briefing note, not an essay. Seven fixed headings in
# a fixed order, so it can be skim-read the same way every time, and a hard
# word budget so it stays that way.
FEEDBACK_SECTIONS = (
    "Overall Assessment",
    "Strengths",
    "Missing Required Skills",
    "Preferred Skills",
    "Eligibility Issues",
    "Weak Resume Sections",
    "Final Recommendation",
)

FEEDBACK_MIN_WORDS = 150
FEEDBACK_MAX_WORDS = 250


def _plural(count: int, singular: str, plural: str = "") -> str:
    return singular if count == 1 else (plural or singular + "s")


def _verdict(ats_score: float) -> str:
    if ats_score >= 80:
        return "a strong match"
    if ats_score >= 65:
        return "a solid match"
    if ats_score >= 50:
        return "a partial match"
    return "a weak match"


def _word_count(sections: list[dict[str, str]]) -> int:
    return sum(len(item["text"].split()) for item in sections)


def _trim_to_budget(sections: list[dict[str, str]], budget: int = FEEDBACK_MAX_WORDS) -> list[dict[str, str]]:
    """Keep the briefing inside its word budget.

    Trims from the back of the longest section first, so the headings all
    survive and the earliest, most important ones are never cut.
    """
    while _word_count(sections) > budget:
        longest = max(sections, key=lambda item: len(item["text"].split()))
        words = longest["text"].split()
        if len(words) <= 8:
            break
        longest["text"] = " ".join(words[:-4]).rstrip(",;") + "."
    return sections


def build_recruiter_summary(
    report: dict[str, Any],
    section_scores: dict[str, float],
    ats_score: float,
    components: dict[str, float],
) -> list[dict[str, str]]:
    """Seven short paragraphs a recruiter can read in under a minute."""
    skills = _skill_feedback(report)
    requirement_report = report.get("requirement_report", {}) or {}
    counts = requirement_report.get("level_counts", {}) or {}
    eligibility = report.get("eligibility", {}) or {}
    checks = eligibility.get("checks", []) or []
    inferred_evidence = report.get("resume_inferred_evidence", {}) or {}
    _strengths, weak_sections = _best_and_weakest_sections(
        section_scores, _applicable_sections(report)
    )

    evaluated = int(requirement_report.get("evaluated", 0) or 0)
    strong = int(counts.get("Strong Match", 0))
    moderate = int(counts.get("Moderate Match", 0))

    # 1. Overall Assessment
    drivers = (
        f"Required-skill coverage is {components.get('skill', 0.0)}% and keyword "
        f"overlap {components.get('keyword', 0.0)}%."
    )
    if evaluated:
        overall = (
            f"Scores {ats_score}% against this posting, {_verdict(ats_score)}. "
            f"Of {evaluated} stated requirement(s), {strong} are strongly evidenced "
            f"and {moderate} partially. {drivers}"
        )
    else:
        named = len(report.get("required_skills", []) or [])
        overall = (
            f"Scores {ats_score}% against this posting, {_verdict(ats_score)}. "
            "The posting states no checkable requirements, so this reflects skill "
            f"and keyword overlap only. {drivers} "
            f"Only {named} named {_plural(named, 'skill')} could be read from the "
            "posting, so treat this score as indicative rather than decisive."
        )

    # 2. Strengths
    if skills["matched"]:
        strengths = f"Directly evidences {_join_phrases(sorted(skills['matched']), limit=5)}."
        credited = [skill for skill in skills["partial"] if skill in inferred_evidence]
        if credited:
            strengths += f" A further {len(credited)} implied by related experience."
        strong_items = (requirement_report.get("by_level", {}) or {}).get("Strong Match", [])
        if strong_items:
            example = str(strong_items[0].get("evidence") or "").strip()
            if example:
                strengths += f' Strongest evidence: "{example[:90]}".'
    else:
        strengths = "No skill the posting names appears directly in the resume."

    # 3. Missing Required Skills - technical gaps stated as the priority, with
    #    behavioural ones flagged separately rather than lumped in with them.
    technical_gaps = sorted(skills["missing_required_technical"])
    soft_gaps = sorted(skills["missing_required_soft"])
    if technical_gaps:
        missing = (
            f"Technical: {_join_phrases(technical_gaps, limit=5)} not evidenced. "
            "These are the priority gap."
        )
        if soft_gaps:
            missing += f" Soft: {_join_phrases(soft_gaps, limit=3)} also unstated."
    elif soft_gaps:
        missing = (
            "All technical requirements are evidenced. "
            f"{_join_phrases(soft_gaps, limit=3)} appear as stated requirements but are "
            "not written into the resume."
        )
    else:
        missing = "None. Every skill the posting requires is evidenced."

    # 4. Preferred Skills
    if skills["missing_preferred"]:
        count = len(skills["missing_preferred"])
        preferred = (
            f"Missing the optional {_plural(count, 'extra')} "
            f"{_join_phrases(sorted(skills['missing_preferred']), limit=4)}, which "
            f"{_plural(count, 'carries', 'carry')} little weight and should not block progression."
        )
    else:
        preferred = "No preferred-only skills are missing."

    # 5. Eligibility Issues
    concerns = [check for check in checks if check.get("status") in {"fail", "borderline"}]
    if concerns:
        eligibility_text = " ".join(
            f"{check.get('title')}: {check.get('detail', '')}" for check in concerns[:2]
        ).strip()
    elif checks:
        cleared = "; ".join(
            f"{check.get('title')} {check.get('candidate_value')} against {check.get('job_value')}"
            for check in checks
            if check.get("candidate_value") and check.get("job_value")
        )
        eligibility_text = "All checks clear the stated bar."
        if cleared:
            eligibility_text += f" {cleared}."
    else:
        eligibility_text = "Not assessed - the documents state no eligibility criteria."

    # 6. Weak Resume Sections
    if weak_sections:
        shown = weak_sections[:3]
        weak_text = (
            f"{_join_phrases(shown)} {_plural(len(shown), 'is', 'are')} the thinnest. "
            "Specific, measurable evidence there would lift the score fastest."
        )
    else:
        weak_text = "No section scores poorly; the resume is evenly developed."

    # 7. Final Recommendation
    blocking = eligibility.get("blocking", []) or []
    if blocking:
        final = (
            f"Confirm {_join_phrases(blocking, limit=2)} before progressing; "
            "the technical fit is otherwise assessed above."
        )
    elif skills["missing_required"]:
        final = "Worth a screening call if the missing requirements can be evidenced in conversation."
    elif ats_score >= 65:
        final = "Recommend progressing to interview."
    else:
        final = "Hold unless the shortlist is thin; the alignment is limited."

    # One concrete lever, so the briefing ends on something actionable rather
    # than a verdict alone.
    next_step = (report.get("_top_recommendation") or "").strip()
    if next_step:
        final = f"{final} Highest-value fix: {next_step[0].lower()}{next_step[1:]}"

    texts = [overall, strengths, missing, preferred, eligibility_text, weak_text, final]
    sections = [
        {"title": title, "text": text}
        for title, text in zip(FEEDBACK_SECTIONS, texts)
    ]
    # A thin posting or a thin resume genuinely leaves less to say. Rather than
    # padding, add the next most useful fact a recruiter would ask for, in
    # order of usefulness, until the briefing reaches a readable length.
    uncovered = requirement_report.get("uncovered", []) or []
    missing_keywords = (report.get("keyword_match", {}) or {}).get("missing", []) or []
    extras: list[str] = []
    if uncovered:
        extras.append(f'Unanswered: "{uncovered[0][:80]}".')
    if missing_keywords:
        extras.append(
            "Wording the resume never uses: " + ", ".join(missing_keywords[:5]) + "."
        )
    extras.append(
        "Read alongside the eligibility panel and the responsibility breakdown "
        "before making a shortlisting decision."
    )
    for extra in extras:
        if _word_count(sections) >= FEEDBACK_MIN_WORDS:
            break
        sections[0]["text"] += f" {extra}"

    return _trim_to_budget(sections)


def _recruiter_feedback(
    report: dict[str, Any],
    section_scores: dict[str, float],
    ats_score: float,
    components: dict[str, float],
) -> str:
    """Flat-text form of the briefing, kept for callers that want one string."""
    sections = build_recruiter_summary(report, section_scores, ats_score, components)
    return " ".join(f"{item['title']}: {item['text']}" for item in sections)


PRIORITY_HIGH = "High Priority"
PRIORITY_MEDIUM = "Medium Priority"
PRIORITY_OPTIONAL = "Optional"

PRIORITY_ORDER = (PRIORITY_HIGH, PRIORITY_MEDIUM, PRIORITY_OPTIONAL)

PRIORITY_ICONS = {
    PRIORITY_HIGH: "\U0001F525",
    PRIORITY_MEDIUM: "\U0001F7E1",
    PRIORITY_OPTIONAL: "\U0001F7E2",
}


def build_recommendations(report: dict[str, Any]) -> list[dict[str, str]]:
    """Fixes with a priority attached, in the order they are worth doing.

    High     - required technical gaps and anything blocking a screen. These
               are the fixes that change whether the resume passes at all.
    Medium   - unanswered requirements, wording, weak sections, soft-skill
               requirements. Real improvements, none of them disqualifying.
    Optional - preferred extras and general polish.

    Returns a list of {"text", "priority"} rather than bare strings, so the
    report can group them instead of printing one long undifferentiated list.
    """
    skills_config = report.get("skills_config", {})
    recommendation_map = skills_config.get("recommendations", {})
    section_scores = _section_scores(report)
    requirement_report = report.get("requirement_report", {}) or {}
    skills = _skill_feedback(report)
    inferred_evidence = report.get("resume_inferred_evidence", {}) or {}
    eligibility = report.get("eligibility", {}) or {}

    recommendations: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(message: str, priority: str) -> None:
        if message and message not in seen:
            recommendations.append({"text": message, "priority": priority})
            seen.add(message)

    # --- High: required technical skills the resume does not evidence.
    for skill in skills["missing_required_technical"][:5]:
        add(
            recommendation_map.get(skill, f"Add {skill} experience or projects - the posting requires it."),
            PRIORITY_HIGH,
        )

    # Required skills the analyzer only inferred should be written out, because
    # a keyword-driven ATS will not make the same leap this one does.
    implied_required = [skill for skill in skills["partial"] if skill in inferred_evidence][:2]
    for skill in implied_required:
        add(
            f"State {skill} explicitly. It is currently only implied by your other "
            "experience, and keyword-based filters will not infer it.",
            PRIORITY_HIGH,
        )

    # --- High / Medium: the experience gap, graded by how large it is.
    for check in eligibility.get("checks", []) or []:
        if check.get("key") != "experience":
            continue
        if check.get("status") == "fail":
            add(
                "Close the experience gap on paper: add internships, freelance work, "
                "and academic or personal projects with dates so your total years are visible.",
                PRIORITY_HIGH,
            )
        elif check.get("status") == "borderline":
            add(
                "Make your total years of experience explicit and unbroken, so the "
                "shortfall against the posting reads as small as it is.",
                PRIORITY_MEDIUM,
            )
        elif check.get("status") == "unknown" and check.get("job_value") != "Not stated":
            add(
                "State your total years of experience in the summary line - the posting "
                "sets a minimum and the resume never gives a figure.",
                PRIORITY_MEDIUM,
            )

    # --- Medium: requirements never answered, wording, weak sections.
    for requirement in (requirement_report.get("uncovered", []) or [])[:3]:
        add(f'Address this requirement explicitly: "{requirement[:90]}".', PRIORITY_MEDIUM)

    missing_keywords = report.get("keyword_match", {}).get("missing", []) or report.get("missing_keywords", [])
    if missing_keywords:
        add(
            f"Use more of the posting's own wording, such as {', '.join(missing_keywords[:6])}.",
            PRIORITY_MEDIUM,
        )

    for skill in skills["missing_required_soft"][:2]:
        add(
            f"Evidence {skill} through a bullet describing a situation, rather than "
            "listing it as an adjective.",
            PRIORITY_MEDIUM,
        )

    if section_scores.get("technical_skills", 0) < 70:
        add(
            "Strengthen the technical skills section with the exact tools, libraries, and platforms named in the posting.",
            PRIORITY_MEDIUM,
        )
    if section_scores.get("work_experience", 0) < 70:
        add("Rewrite experience bullets to lead with the outcome and a number, not the task.", PRIORITY_MEDIUM)
    if section_scores.get("projects", 0) < 60:
        add(
            "Add a project that demonstrates the missing skills in action, with the business problem it solved.",
            PRIORITY_MEDIUM,
        )
    if section_scores.get("professional_summary", 0) < 60:
        add("Tailor the summary to mirror the posting's language, target role, and tools.", PRIORITY_MEDIUM)
    if section_scores.get("education", 0) < 50:
        add(
            "Add relevant coursework, certifications, or training that support the role requirements.",
            PRIORITY_MEDIUM,
        )
    if section_scores.get("soft_skills", 0) < 50:
        add(
            "Show collaboration and stakeholder communication through bullets rather than a list of adjectives.",
            PRIORITY_OPTIONAL,
        )

    # --- Optional: preferred extras and general polish.
    for skill in skills["missing_preferred"][:2]:
        add(
            f"{skill} is listed as preferred, not required - worth adding if you have any exposure.",
            PRIORITY_OPTIONAL,
        )

    for item in (
        "Add measurable achievements and outcomes to your bullet points.",
        "Include project links, GitHub, or portfolio evidence where relevant.",
        "Add internship, freelance, or academic project details if commercial experience is limited.",
    ):
        if len(recommendations) >= 4:
            break
        add(item, PRIORITY_OPTIONAL)

    # Capped at eight: enough to fill three priority groups without turning
    # the panel back into the long list this change set out to break up.
    return recommendations[:8]


def group_recommendations(recommendations: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Recommendations bucketed by priority, highest first, empties dropped."""
    groups: list[dict[str, Any]] = []
    for priority in PRIORITY_ORDER:
        items = [item["text"] for item in recommendations if item["priority"] == priority]
        if items:
            groups.append(
                {"priority": priority, "icon": PRIORITY_ICONS[priority], "items": items}
            )
    return groups


def build_ats_report(report: dict[str, Any]) -> dict[str, Any]:
    keyword_score = report["keyword_match"]["score"]
    skill_score = report["skill_match"]["score"]
    similarity_score = report["similarity"]
    section_scores = _section_scores(report)

    raw_composite, components = _composite_score(report, section_scores)
    ats_score = calibrate_score(raw_composite)

    if ats_score <= 0 and (keyword_score or skill_score or similarity_score):
        ats_score = _weighted_score(keyword_score, skill_score, similarity_score)

    matched_keywords = report.get("matched_keywords", [])
    missing_keywords = report.get("missing_keywords", [])
    skills = _skill_feedback(report)
    matched_skills = skills["matched"]
    partial_skills = skills["partial"]
    missing_skills = skills["missing_required"]
    missing_preferred_skills = skills["missing_preferred"]

    matched_skills_count = len(matched_skills)
    partial_skills_count = len(partial_skills)
    missing_skills_count = len(missing_skills)

    section_analysis = report.get("section_analysis", {})
    technical_score = section_scores.get("technical_skills", 0.0)
    experience_score = section_scores.get("work_experience", 0.0)
    projects_score = section_scores.get("projects", 0.0)
    summary_score = section_scores.get("professional_summary", 0.0)
    education_score = section_scores.get("education", 0.0)
    soft_skills_score = section_scores.get("soft_skills", 0.0)

    responsibility_matches = report.get("responsibility_matches", [])
    partial_responsibility_matches = report.get("partial_responsibility_matches", [])
    recommendations = build_recommendations(report)
    # Handed to the summary builder so the briefing can close on the single
    # highest-value fix without recomputing the whole recommendation list.
    report["_top_recommendation"] = recommendations[0]["text"] if recommendations else ""
    recruiter_summary = build_recruiter_summary(report, section_scores, ats_score, components)
    recruiter_feedback = " ".join(
        f"{item['title']}: {item['text']}" for item in recruiter_summary
    )

    return {
        "ats_score": ats_score,
        "score_label": _score_label(ats_score),
        "keyword_score": keyword_score,
        "skill_score": skill_score,
        "similarity_score": similarity_score,
        "matched_keywords": matched_keywords[:20],
        "missing_keywords": missing_keywords[:20],
        "matched_skills": matched_skills[:20],
        "partial_skills": partial_skills[:20],
        # Required-only, so a missing must-have is never shown next to an
        # optional extra as if they carried the same consequence.
        "missing_skills": missing_skills[:20],
        # Task 4: required gaps split by kind.
        "missing_required_technical_skills": skills["missing_required_technical"][:20],
        "missing_required_soft_skills": skills["missing_required_soft"][:20],
        "required_technical_skills": report.get("required_technical_skills", [])[:30],
        "required_soft_skills": report.get("required_soft_skills", [])[:20],
        "missing_preferred_skills": missing_preferred_skills[:20],
        "missing_inferred_skills": skills["missing_inferred"][:20],
        "required_skills": report.get("required_skills", [])[:30],
        "preferred_skills": report.get("preferred_skills", [])[:20],
        "skills_found": matched_skills_count,
        "skills_partial": partial_skills_count,
        "skills_missing": missing_skills_count,
        "skills_missing_preferred": len(missing_preferred_skills),
        "section_scores": section_scores,
        "technical_skills_score": technical_score,
        "experience_score": experience_score,
        "projects_score": projects_score,
        "summary_score": summary_score,
        "education_score": education_score,
        "soft_skills_score": soft_skills_score,
        "section_analysis": section_analysis,
        "responsibility_matches": responsibility_matches[:10],
        "partial_responsibility_matches": partial_responsibility_matches[:10],
        # Every responsibility, graded Strong / Moderate / Weak / Missing with
        # the evidence behind each verdict.
        "responsibility_breakdown": report.get("responsibility_breakdown", {}),
        "responsibility_level_counts": report.get("responsibility_level_counts", {}),
        "recruiter_feedback": recruiter_feedback,
        # Same content as recruiter_feedback, split into its seven headings so
        # the template can lay it out instead of printing one long paragraph.
        "recruiter_summary": recruiter_summary,
        "recruiter_feedback_words": len(recruiter_feedback.split()),
        # Flat list of strings, kept so any existing consumer keeps working.
        "recommendations": [item["text"] for item in recommendations],
        # Task 8: the same fixes grouped High / Medium / Optional.
        "recommendation_groups": group_recommendations(recommendations),
        # Task 5: Education as a verdict, not a percentage.
        "education_verdict": build_education_verdict(report),
        # Task 7: shortlisting decision, driven by screening criteria rather
        # than by the ATS score.
        "recruiter_decision": build_recruiter_decision(report),
        # Eligibility screening. Reported next to the score, never inside it -
        # location and work mode are hard filters, not quality signals.
        "eligibility": report.get("eligibility", {}),
        "education_assessment": report.get("education_assessment", {}),
        # Additive diagnostics; the template ignores unknown keys.
        "score_components": components,
        "raw_composite_score": components.get("raw_composite", 0.0),
        "requirement_report": report.get("requirement_report", {}),
        "inferred_skills": report.get("resume_inferred_skills", []),
        "raw": report,
    }