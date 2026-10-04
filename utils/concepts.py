"""Concept inference and lexical equivalence.

This module is additive: nothing in the original pipeline is replaced. It
supplies two capabilities the analyzer previously lacked.

1. Concept inference. A skill written in a resume is evidence for broader
   concepts that are rarely spelled out. "Power BI" is evidence of dashboard
   development, data visualization, business intelligence and reporting, even
   when none of those phrases appear. The implication graph is declared in
   skills.json under each entry's ``implies`` key and closed transitively here.

2. Lexical equivalence for responsibilities. "Develop dashboards" and "Built
   interactive Power BI dashboards" describe the same duty. Embeddings alone
   score this inconsistently, so we additionally reduce each phrase to an
   (action, object) shape using synonym groups and compare those directly.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

# --------------------------------------------------------------------------
# Action verb groups. Every verb in a group is treated as the same intent.
# --------------------------------------------------------------------------
ACTION_GROUPS: dict[str, tuple[str, ...]] = {
    "build": (
        "build", "built", "develop", "developed", "create", "created", "design",
        "designed", "implement", "implemented", "produce", "produced", "deliver",
        "delivered", "construct", "constructed", "engineer", "engineered",
        "author", "authored", "prototype", "prototyped", "establish",
        "established", "set up", "stand up", "ship", "shipped", "launch",
        "launched", "write", "wrote", "writing", "perform", "performed",
        "conduct", "conducted", "execute", "executed", "undertake",
        "undertook", "carry", "carried", "provide", "provided",
    ),
    "analyze": (
        "analyze", "analyse", "analyzed", "analysed", "examine", "examined",
        "investigate", "investigated", "evaluate", "evaluated", "assess",
        "assessed", "interpret", "interpreted", "study", "studied", "explore",
        "explored", "review", "reviewed", "measure", "measured", "profile",
        "profiled",
    ),
    "maintain": (
        "maintain", "maintained", "manage", "managed", "own", "owned",
        "administer", "administered", "operate", "operated", "support",
        "supported", "monitor", "monitored", "oversee", "oversaw", "handle",
        "handled", "run", "ran",
    ),
    "improve": (
        "improve", "improved", "optimize", "optimise", "optimized", "optimised",
        "enhance", "enhanced", "refactor", "refactored", "streamline",
        "streamlined", "increase", "increased", "reduce", "reduced", "cut",
        "accelerate", "accelerated", "tune", "tuned", "scale", "scaled",
    ),
    "automate": (
        "automate", "automated", "schedule", "scheduled", "orchestrate",
        "orchestrated", "script", "scripted", "streamline", "streamlined",
    ),
    "collaborate": (
        "collaborate", "collaborated", "partner", "partnered", "coordinate",
        "coordinated", "liaise", "liaised", "work with", "worked with",
        "engage", "engaged", "align", "aligned", "consult", "consulted",
    ),
    "communicate": (
        "present", "presented", "communicate", "communicated", "report",
        "reported", "share", "shared", "explain", "explained", "document",
        "documented", "visualize", "visualise", "visualized", "visualised",
        "summarize", "summarise", "summarized", "summarised", "translate",
        "translated",
    ),
    "lead": (
        "lead", "led", "mentor", "mentored", "supervise", "supervised", "guide",
        "guided", "train", "trained", "coach", "coached", "direct", "directed",
        "drive", "drove", "spearhead", "spearheaded",
    ),
    "clean": (
        "clean", "cleaned", "cleanse", "cleansed", "prepare", "prepared",
        "preprocess", "preprocessed", "transform", "transformed", "wrangle",
        "wrangled", "normalize", "normalise", "normalized", "normalised",
        "validate", "validated", "reconcile", "reconciled",
    ),
    "extract": (
        "extract", "extracted", "query", "queried", "retrieve", "retrieved",
        "pull", "pulled", "collect", "collected", "gather", "gathered",
        "ingest", "ingested", "source", "sourced", "load", "loaded",
    ),
    "model": (
        "model", "modeled", "modelled", "forecast", "forecasted", "predict",
        "predicted", "train", "trained", "classify", "classified", "cluster",
        "clustered", "segment", "segmented", "estimate", "estimated",
    ),
    "test": (
        "test", "tested", "verify", "verified", "audit", "audited", "check",
        "checked", "qa", "troubleshoot", "troubleshot", "debug", "debugged",
        "diagnose", "diagnosed", "resolve", "resolved",
    ),
}

# Resumes nominalise what job descriptions verbalise: a JD says "analyse data"
# where a resume says "data analysis". Without this map the two never align.
NOMINALIZATIONS: dict[str, str] = {
    "analysis": "analyze",
    "analytics": "analyze",
    "analytical": "analyze",
    "evaluation": "analyze",
    "assessment": "analyze",
    "development": "build",
    "creation": "build",
    "implementation": "build",
    "construction": "build",
    "delivery": "build",
    "optimization": "improve",
    "optimisation": "improve",
    "improvement": "improve",
    "enhancement": "improve",
    "automation": "automate",
    "orchestration": "automate",
    "maintenance": "maintain",
    "administration": "maintain",
    "monitoring": "maintain",
    "collaboration": "collaborate",
    "coordination": "collaborate",
    "cleaning": "clean",
    "cleansing": "clean",
    "preparation": "clean",
    "preprocessing": "clean",
    "transformation": "clean",
    "wrangling": "clean",
    "extraction": "extract",
    "ingestion": "extract",
    "modeling": "model",
    "modelling": "model",
    "forecasting": "model",
    "prediction": "model",
    "classification": "model",
    "segmentation": "model",
    "validation": "test",
    "verification": "test",
    "troubleshooting": "test",
    "mentoring": "lead",
    "supervision": "lead",
    "presentation": "communicate",
    "documentation": "communicate",
    "visualization": "communicate",
    "visualisation": "communicate",
}

_ACTION_LOOKUP: dict[str, str] = {}
for _group, _verbs in ACTION_GROUPS.items():
    for _verb in _verbs:
        _ACTION_LOOKUP.setdefault(_verb, _group)

# --------------------------------------------------------------------------
# Object noun groups. Different words for the same deliverable.
# --------------------------------------------------------------------------
OBJECT_GROUPS: dict[str, tuple[str, ...]] = {
    "dashboard": ("dashboard", "dashboards", "dashboarding", "visualization",
                  "visualizations", "visualisation", "visualisations", "chart",
                  "charts", "scorecard", "scorecards", "visual", "visuals"),
    "report": ("report", "reports", "reporting", "summary", "summaries",
               "deliverable", "deliverables", "presentation", "presentations",
               "deck", "decks"),
    "data": ("data", "dataset", "datasets", "database", "databases", "records",
             "information", "table", "tables", "warehouse", "lakehouse"),
    "insight": ("insight", "insights", "finding", "findings", "recommendation",
                "recommendations", "conclusion", "conclusions", "trend",
                "trends", "pattern", "patterns"),
    "pipeline": ("pipeline", "pipelines", "etl", "elt", "workflow", "workflows",
                 "job", "jobs", "ingestion", "integration"),
    "model": ("model", "models", "algorithm", "algorithms", "classifier",
              "classifiers", "forecast", "forecasts", "prediction",
              "predictions"),
    "query": ("query", "queries", "sql", "script", "scripts", "procedure",
              "procedures"),
    "metric": ("metric", "metrics", "kpi", "kpis", "measure", "measures",
               "performance", "target", "targets", "benchmark", "benchmarks"),
    "process": ("process", "processes", "procedure", "procedures", "system",
                "systems", "operation", "operations", "workflow"),
    "requirement": ("requirement", "requirements", "specification",
                    "specifications", "need", "needs", "brief", "scope"),
    "quality": ("quality", "accuracy", "integrity", "consistency", "validation",
                "governance", "compliance"),
    "application": ("application", "applications", "app", "apps", "service",
                    "services", "api", "apis", "platform", "tool", "tools",
                    "solution", "solutions", "product"),
}

# Audience nouns describe *who* a duty serves, not *what* is produced. Almost
# every bullet in every resume mentions an audience, so treating these as
# deliverables makes unrelated work look aligned ("designed brand identities
# for clients" would otherwise match "develop dashboards for stakeholders").
# They are scored separately at a much lower weight.
CONTEXT_GROUPS: dict[str, tuple[str, ...]] = {
    "stakeholder": ("stakeholder", "stakeholders", "client", "clients",
                    "customer", "customers", "leadership", "management",
                    "executive", "executives", "business", "team", "teams",
                    "partner", "partners", "user", "users", "audience",
                    "department", "departments", "organization", "organisation"),
}

_OBJECT_LOOKUP: dict[str, str] = {}
for _group, _nouns in OBJECT_GROUPS.items():
    for _noun in _nouns:
        _OBJECT_LOOKUP.setdefault(_noun, _group)

_CONTEXT_LOOKUP: dict[str, str] = {}
for _group, _nouns in CONTEXT_GROUPS.items():
    for _noun in _nouns:
        _CONTEXT_LOOKUP.setdefault(_noun, _group)

# Words that carry no matching signal inside a responsibility phrase.
_PHRASE_NOISE = {
    "a", "an", "the", "and", "or", "for", "with", "to", "of", "in", "on", "at",
    "by", "from", "as", "that", "this", "these", "those", "our", "their", "its",
    "we", "you", "your", "will", "must", "should", "can", "able", "ability",
    "strong", "excellent", "good", "solid", "proven", "demonstrated", "hands",
    "experience", "experienced", "knowledge", "skills", "skill", "years",
    "year", "plus", "etc", "including", "include", "includes", "such",
    "various", "multiple", "new", "existing", "relevant", "related", "using",
    "used", "use", "across", "within", "into", "over", "under", "up", "well",
    "also", "other", "others", "who", "which", "what", "when", "while", "be",
    "is", "are", "was", "were", "been", "being", "have", "has", "had", "do",
    "does", "did", "not", "no", "all", "any", "some", "more", "most", "least",
    "role", "candidate", "position", "job", "company", "successful", "ideal",
    "responsibilities", "requirements", "qualifications", "preferred",
    "required", "desirable", "nice", "bonus",
}


def _tokenize(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9+#]+", (text or "").lower()) if token]


# --------------------------------------------------------------------------
# Concept graph
# --------------------------------------------------------------------------
def build_implication_map(skills_config: dict[str, Any]) -> dict[str, list[str]]:
    """Return skill name -> directly implied concept names."""
    implications: dict[str, list[str]] = {}
    for _category, entries in (skills_config.get("categories", {}) or {}).items():
        for entry in entries:
            if isinstance(entry, str):
                continue
            name = str(entry.get("name", "")).strip()
            implied = [str(item).strip() for item in entry.get("implies", []) if str(item).strip()]
            if name and implied:
                implications[name] = implied
    return implications


def _close_transitively(name: str, implications: dict[str, list[str]], depth: int = 3) -> set[str]:
    seen: set[str] = set()
    frontier = list(implications.get(name, []))
    for _ in range(depth):
        if not frontier:
            break
        next_frontier: list[str] = []
        for concept in frontier:
            if concept in seen or concept == name:
                continue
            seen.add(concept)
            next_frontier.extend(implications.get(concept, []))
        frontier = next_frontier
    return seen


def infer_concepts(
    present_skills: Iterable[str],
    skills_config: dict[str, Any],
) -> dict[str, list[str]]:
    """Map each present skill to the concepts it is evidence for.

    Only concepts not already directly present are returned, so the caller can
    distinguish stated skills from inferred ones.
    """
    implications = build_implication_map(skills_config)
    present = {skill for skill in present_skills if skill}
    inferred: dict[str, list[str]] = {}

    for skill in present:
        for concept in _close_transitively(skill, implications):
            if concept in present:
                continue
            inferred.setdefault(concept, []).append(skill)

    return {concept: sorted(set(sources)) for concept, sources in inferred.items()}


# --------------------------------------------------------------------------
# Responsibility shapes
# --------------------------------------------------------------------------
def phrase_shape(text: str) -> dict[str, set[str]]:
    """Reduce a phrase to canonical actions, deliverables, audience and keywords."""
    tokens = _tokenize(text)
    actions: set[str] = set()
    objects: set[str] = set()
    context: set[str] = set()
    keywords: set[str] = set()

    for token in tokens:
        if token in _ACTION_LOOKUP:
            actions.add(_ACTION_LOOKUP[token])
            continue
        if token in _OBJECT_LOOKUP:
            # Deliberately not also added to keywords. The group already
            # captures the term, and keeping the raw token would make
            # "datasets" and "data" miss each other in the keyword comparison.
            objects.add(_OBJECT_LOOKUP[token])
            continue
        if token in _CONTEXT_LOOKUP:
            context.add(_CONTEXT_LOOKUP[token])
            continue
        if token in NOMINALIZATIONS:
            actions.add(NOMINALIZATIONS[token])
            continue
        if token in _PHRASE_NOISE or len(token) < 3:
            continue
        keywords.add(token)

    return {
        "actions": actions,
        "objects": objects,
        "context": context,
        "keywords": keywords,
    }


def _coverage(job_side: set[str], resume_side: set[str]) -> float:
    """How much of the job's requirement the resume covers.

    Asymmetric on purpose. A resume bullet is usually richer than the job line
    it satisfies ("built interactive Power BI dashboards tracking sales" vs
    "develop dashboards"), and Jaccard punishes that extra detail. What matters
    is whether the job's terms are present, not whether the resume adds more.
    """
    if not job_side:
        return 0.0
    return len(job_side & resume_side) / len(job_side)


def lexical_alignment(job_phrase: str, resume_phrase: str) -> float:
    """Score 0..1 for how well a resume phrase satisfies a job responsibility.

    The deliverable dominates: "develop dashboards" is satisfied by "built
    Power BI dashboards" because both concern dashboards. Shared audience alone
    ("for clients") is near-worthless, since nearly every bullet has one.
    """
    job = phrase_shape(job_phrase)
    resume = phrase_shape(resume_phrase)

    object_cover = _coverage(job["objects"], resume["objects"])
    keyword_cover = _coverage(job["keywords"], resume["keywords"])
    action_cover = _coverage(job["actions"], resume["actions"])
    context_cover = _coverage(job["context"], resume["context"])

    # Audience agreement on its own never constitutes a duty match.
    if object_cover == 0.0 and keyword_cover == 0.0:
        return 0.0

    score = (
        (0.55 * object_cover)
        + (0.25 * keyword_cover)
        + (0.15 * action_cover)
        + (0.05 * context_cover)
    )

    # Same deliverable and same verb intent is a genuine duty match; reward it
    # so terse job phrasing is not penalised against detailed resume bullets.
    if object_cover > 0 and action_cover > 0:
        score = min(1.0, score + 0.20)

    return round(min(1.0, score), 4)


def skill_evidence_bonus(
    job_phrase: str,
    resume_skills: set[str],
    skills_config: dict[str, Any],
) -> float:
    """Credit when a job duty names a concept the resume proves through tools.

    "Develop dashboards" is satisfied by a resume holding Power BI, because
    Power BI implies Dashboard Development. Comparison is on phrase shapes
    rather than substrings, so "Dashboard Development" matches the phrasing
    "develop dashboards".

    The deliverable noun carries the decision: any verb will do, but the thing
    produced must line up. When a duty has no deliverable at all ("strong
    analytical skills") the action intent may stand in for it, but only if the
    phrase also has no unexplained keywords -- otherwise "manage brand
    campaigns" would be satisfied by any concept that merely implies managing.
    """
    if not resume_skills:
        return 0.0

    inferred = set(infer_concepts(resume_skills, skills_config))
    covered = set(resume_skills) | inferred
    if not covered:
        return 0.0

    phrase = phrase_shape(job_phrase)
    concept_shapes = [phrase_shape(name) for name in covered]

    if phrase["objects"]:
        for shape in concept_shapes:
            if phrase["objects"] <= shape["objects"]:
                return 1.0
        return 0.0

    if phrase["keywords"]:
        for shape in concept_shapes:
            if phrase["keywords"] <= shape["keywords"]:
                return 1.0
        return 0.0

    if phrase["actions"]:
        for shape in concept_shapes:
            if phrase["actions"] <= shape["actions"]:
                return 1.0

    return 0.0


def is_meaningful_requirement(text: str) -> bool:
    """Filter out boilerplate lines that should not count as requirements."""
    cleaned = (text or "").strip()
    if len(cleaned.split()) < 3:
        return False
    shape = phrase_shape(cleaned)
    return bool(shape["objects"] or shape["keywords"])


# A recruiter credits communication because a bullet says "presented findings
# to stakeholders", not because the resume contains the word "communication".
# These action intents are strong enough evidence of the corresponding skill
# when they appear in a substantive bullet.
ACTION_EVIDENCE: dict[str, str] = {
    "communicate": "Communication",
    "collaborate": "Teamwork",
    "lead": "Leadership",
    "analyze": "Analytical Thinking",
    "test": "Problem Solving",
    "improve": "Process Improvement",
    "automate": "Automation",
    "clean": "Data Cleaning",
}

# Intents that only count as evidence alongside a real deliverable, to avoid
# crediting a skill from an incidental verb.
_REQUIRES_DELIVERABLE = {"improve", "automate", "clean", "analyze"}


def infer_demonstrated_skills(text: str) -> dict[str, list[str]]:
    """Skills a document demonstrates through described actions.

    Returns skill name -> the lines that evidence it. Only substantive lines
    are considered, so headings and one-word fragments cannot create evidence.
    """
    demonstrated: dict[str, list[str]] = {}

    for raw_line in (text or "").replace("\r", "\n").split("\n"):
        line = raw_line.strip(" -•\t")
        if not is_meaningful_requirement(line):
            continue

        shape = phrase_shape(line)
        if not shape["actions"]:
            continue

        has_deliverable = bool(shape["objects"])
        for action in shape["actions"]:
            skill = ACTION_EVIDENCE.get(action)
            if not skill:
                continue
            if action in _REQUIRES_DELIVERABLE and not has_deliverable:
                continue
            demonstrated.setdefault(skill, []).append(line)

    return {skill: lines[:3] for skill, lines in demonstrated.items()}
