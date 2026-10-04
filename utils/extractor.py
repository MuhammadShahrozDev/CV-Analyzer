from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .concepts import infer_concepts, infer_demonstrated_skills
from .preprocessing import normalize_whitespace


def load_skills_config(skills_file: Path) -> dict[str, Any]:
    if not skills_file.exists():
        raise FileNotFoundError(f"Skills config not found: {skills_file}")
    with skills_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_category_name(name: str) -> str:
    return normalize_whitespace(name).strip()


def _normalize_alias(alias: str) -> str:
    alias = alias.strip().lower()
    alias = re.sub(r"c\s*\+\s*\+", "cpp", alias)
    alias = re.sub(r"c#", "c sharp", alias)
    alias = re.sub(r"[+/,_-]", " ", alias)
    alias = re.sub(r"[^a-z0-9\s]", " ", alias)
    return normalize_whitespace(alias)


def normalize_skill_text(text: str) -> str:
    return _normalize_alias(text)


def _iter_skill_entries(skills_config: dict[str, Any]):
    categories = skills_config.get("categories", {})
    for category_name, values in categories.items():
        for entry in values:
            if isinstance(entry, str):
                canonical_name = normalize_whitespace(entry)
                aliases = [canonical_name]
            else:
                canonical_name = normalize_whitespace(str(entry.get("name", "")))
                aliases = [str(alias) for alias in entry.get("aliases", [])]
                aliases.append(canonical_name)

            if canonical_name:
                cleaned_aliases = sorted({alias for alias in (_normalize_alias(item) for item in aliases) if alias}, key=str.lower)
                yield _normalize_category_name(category_name), {"name": canonical_name, "aliases": cleaned_aliases}


def _alias_tokens(alias: str) -> list[str]:
    return [token for token in alias.split() if token]


# Tokens too generic to identify a skill on their own. A multi-word alias whose
# only hit is one of these has not really been found.
_GENERIC_ALIAS_TOKENS = {
    "data", "analysis", "analytics", "management", "development", "developer",
    "engineering", "engineer", "skills", "skill", "tools", "tool", "system",
    "systems", "software", "programming", "language", "languages", "platform",
    "platforms", "science", "scientist", "analyst", "certified", "certificate",
    "certification", "professional", "associate", "advanced", "basic",
    "microsoft", "google", "amazon", "apache", "web", "cloud", "computing",
    "testing", "processing", "modeling", "modelling", "learning",
}


def _has_phrase_match(normalized_text: str, alias: str) -> bool:
    """Whole-word alias match.

    The previous implementation fell back to a raw substring test for
    single-token aliases, which made short acronyms match inside unrelated
    words: "ai" fired on "bilal ahmed", "bi" on "bilal", "ml" on "html".
    Word boundaries are now required in every case.
    """
    alias_tokens = _alias_tokens(alias)
    if not alias_tokens:
        return False
    pattern = r"\b" + r"\s+".join(re.escape(token) for token in alias_tokens) + r"\b"
    return bool(re.search(pattern, normalized_text))


def _singularize(token: str) -> str:
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es") and not token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def line_token_sets(text: str) -> list[set[str]]:
    """Token sets scoped to a single line, used for proximity-aware matching."""
    sets: list[set[str]] = []
    for raw_line in (text or "").replace("\r", "\n").split("\n"):
        normalized = _normalize_alias(raw_line)
        if not normalized:
            continue
        tokens = {token for token in normalized.split() if token}
        tokens |= {_singularize(token) for token in tokens}
        sets.append(tokens)
    return sets


def _partial_alias_match(line_tokens: list[set[str]], alias: str) -> bool:
    """Near-miss detection for multi-word aliases, scoped to a single line.

    Two prior faults are addressed. Any 50% token overlap used to count, so
    "Data Analysis" registered whenever the word "data" appeared. And overlap
    was measured across the whole document, so "PMP" (alias "project
    management professional") fired on a resume whose PROJECTS heading,
    university name and PROFESSIONAL SUMMARY heading were unrelated and pages
    apart.

    A partial hit now requires the alias tokens to co-occur on one line, cover
    most of the alias, and include at least one distinctive token.
    """
    alias_tokens = _alias_tokens(alias)
    if len(alias_tokens) < 2:
        return False

    alias_set = {_singularize(token) for token in alias_tokens}
    distinctive_alias = {token for token in alias_set if token not in _GENERIC_ALIAS_TOKENS}
    if not distinctive_alias:
        return False

    for tokens in line_tokens:
        overlap = alias_set & tokens
        if not overlap:
            continue
        if not (distinctive_alias & tokens):
            continue
        if (len(overlap) / len(alias_set)) >= 0.75:
            return True

    return False


def extract_skills_from_text(text: str, skills_config: dict[str, Any]) -> dict[str, Any]:
    normalized_text = _normalize_alias(text)
    line_tokens = line_token_sets(text)

    matched_skills: list[str] = []
    partial_skills: list[str] = []
    skill_hits: dict[str, dict[str, Any]] = {}
    all_skills: list[str] = []
    category_map: dict[str, list[str]] = {}
    partial_category_map: dict[str, list[str]] = {}
    category_lookup: dict[str, str] = {}

    for category_name, skill_entry in _iter_skill_entries(skills_config):
        canonical_name = skill_entry["name"]
        all_skills.append(canonical_name)
        category_lookup[canonical_name] = category_name
        category_map.setdefault(category_name, [])
        partial_category_map.setdefault(category_name, [])

        matched_alias = None
        status = "missing"

        for alias in skill_entry["aliases"]:
            if _has_phrase_match(normalized_text, alias):
                matched_alias = alias
                status = "matched"
                break
            if _partial_alias_match(line_tokens, alias):
                matched_alias = alias
                status = "partial"

        if status == "matched":
            matched_skills.append(canonical_name)
            category_map[category_name].append(canonical_name)
        elif status == "partial":
            partial_skills.append(canonical_name)
            partial_category_map[category_name].append(canonical_name)

        if status != "missing":
            skill_hits[canonical_name] = {
                "status": status,
                "category": category_name,
                "alias": matched_alias or canonical_name,
            }

    missing_skills = [skill for skill in all_skills if skill not in matched_skills and skill not in partial_skills]

    # Concept inference: a stated tool is evidence for the broader capabilities
    # it requires. Power BI implies dashboard development, visualization,
    # business intelligence and reporting even when none are written out.
    inferred_map = infer_concepts(matched_skills, skills_config)

    # Behavioural evidence: a bullet describing presenting to stakeholders is
    # evidence of communication even though the word never appears. Only
    # skills the catalog knows about are admitted.
    catalog_names = set(all_skills)
    demonstrated_map = infer_demonstrated_skills(text)
    for skill, lines in demonstrated_map.items():
        if skill not in catalog_names or skill in matched_skills:
            continue
        inferred_map.setdefault(skill, []).extend(lines)

    inferred_skills = sorted(inferred_map)

    # Inferred concepts are real evidence, so they must not be reported as
    # absent from the document.
    inferred_set = set(inferred_skills)
    missing_skills = [skill for skill in missing_skills if skill not in inferred_set]

    for concept, sources in inferred_map.items():
        if concept in skill_hits:
            continue
        skill_hits[concept] = {
            "status": "inferred",
            "category": category_lookup.get(concept, "Inferred"),
            "alias": concept,
            "evidence": sources,
        }

    return {
        "matched_skills": matched_skills,
        "partial_skills": partial_skills,
        "missing_skills": missing_skills,
        "matched_categories": {category: values for category, values in category_map.items() if values},
        "partial_categories": {category: values for category, values in partial_category_map.items() if values},
        "skill_hits": skill_hits,
        "all_skills": all_skills,
        # New, additive. Existing consumers are unaffected.
        "inferred_skills": inferred_skills,
        "inferred_evidence": inferred_map,
        "effective_skills": sorted(set(matched_skills) | inferred_set),
    }
