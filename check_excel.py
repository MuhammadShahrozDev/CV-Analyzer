"""Throwaway check: what does the detector make of a workbook?

    python check_excel.py engn-jobs.xlsx

Delete this file once the batch UI exists.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.excel_jobs import detect_columns, inspect_workbook, read_jobs

path = sys.argv[1] if len(sys.argv) > 1 else "engn-jobs.xlsx"

info = inspect_workbook(path)
print(f"sheets      : {info['sheet_names']}")
print(f"has headers : {info['has_headers']}  (header row {info['header_row']})")
print(f"data rows   : {info['row_count']}")

detected = detect_columns(info["columns"])
names = {column["index"]: column["name"] for column in info["columns"]}

print("\nMAPPED COLUMNS")
for field, index in sorted(detected["mapping"].items(), key=lambda item: item[1]):
    label = detected["labels"][field]
    print(f"  {label:24} -> col {index:>2}  '{names[index]}'  (confidence {detected['confidence'][field]})")

if detected["ambiguous"]:
    print("\nAMBIGUOUS - you would be asked to choose:")
    for field, options in detected["ambiguous"].items():
        print(f"  {detected['labels'][field]}: {[option['name'] for option in options]}")

print(f"\nmissing required: {detected['missing_required'] or 'none'}")

if not detected["missing_required"]:
    jobs = read_jobs(path, detected["mapping"], header_row=info["header_row"], limit=3)
    skipped = sum(1 for job in jobs if job.get("skip_reason"))
    print(f"\nfirst {len(jobs)} rows read ({skipped} skipped):")
    for job in jobs:
        role = (job.get("role") or "")[:38]
        company = (job.get("company") or "")[:16]
        print(f"  row {job['row']:>3}: {role:40} {company:18} description {len(job.get('description',''))} chars")