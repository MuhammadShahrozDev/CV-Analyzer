"""Throwaway diagnostic: shows exactly why the experience requirement is wrong.

Put your real job description in jd.txt next to this file, then run:

    python diagnose_experience.py

Delete this file afterwards.
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from utils import eligibility

print("=" * 68)
print("1. WHICH FILE IS ACTUALLY LOADED")
print("=" * 68)
print("   path:", eligibility.__file__)
has_new = hasattr(eligibility, "_vetoed") and hasattr(eligibility, "_fallback_experience")
print("   contains the latest fix:", has_new)
if not has_new:
    print()
    print("   >>> This is an OLD copy. The replacement did not take effect.")
    print("   >>> Check the path above, delete __pycache__ and utils/__pycache__,")
    print("   >>> and make sure you replaced the file at that exact location.")
    raise SystemExit(1)

jd_path = BASE / "jd.txt"
if not jd_path.exists():
    print("\n   jd.txt not found next to this script - create it and paste your JD.")
    raise SystemExit(1)

raw = jd_path.read_text(encoding="utf-8", errors="replace")

print()
print("=" * 68)
print("2. THE TEXT THE PARSER RECEIVES")
print("=" * 68)
try:
    from utils.parser import normalize_job_description
    flat, layout = normalize_job_description(raw)
except Exception as exc:                                   # pragma: no cover
    print("   normalize_job_description failed:", exc)
    layout = raw

print(f"   raw characters: {len(raw)}  ->  after cleaning: {len(layout)}")
print("   lines mentioning years or experience, AS THE PARSER SEES THEM:")
shown = 0
for number, line in enumerate(layout.split("\n"), 1):
    low = line.lower()
    if "year" in low or "experien" in low or "yrs" in low:
        print(f"      [{number:>3}] {line[:150]}")
        shown += 1
if not shown:
    print("      (none - the requirement was removed before parsing)")

print()
print("=" * 68)
print("3. WHAT THE PARSER EXTRACTED")
print("=" * 68)
result = eligibility.extract_experience_requirement(layout)
print("   min_years       :", result["min_years"])
print("   max_years       :", result["max_years"])
print("   preferred_years :", result["preferred_years"])
print("   chosen phrase   :", repr(result["raw"]))
print("   displayed as    :", eligibility.format_required_experience(result))
print("   every required candidate found:")
for item in result.get("required_candidates", []) or [["(none)"]]:
    print("      ", item)
print("   every preferred candidate found:")
for item in result.get("preferred_candidates", []) or [["(none)"]]:
    print("      ", item)
print()
print("Send me section 2 and section 3 of this output.")
