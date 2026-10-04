# Setup

Complete, runnable project. Replace your old folder with this one, or copy the
files over the top of it.

## 1. Create a virtual environment

Windows:

    python -m venv .venv
    .venv\Scripts\activate

macOS / Linux:

    python3 -m venv .venv
    source .venv/bin/activate

## 2. Install dependencies

    pip install -r requirements.txt
    python -m spacy download en_core_web_sm

Optional, only needed to read JavaScript-rendered job pages from a URL:

    playwright install chromium

## 3. Run

    python -m uvicorn app:app --reload --port 8001

Open http://127.0.0.1:8001

## 4. Run the tests (optional)

    python tests_calibration.py   # score ordering and spread across six resumes
    python tests_eligibility.py   # tiering, related degrees, screening, input parity

## What is in this package

Updated in this round:

    app.py                  unified job-description handling, single-parse uploads,
                            safer error page, import cleanup
    utils/eligibility.py    NEW - location, work mode, experience, education screening
    utils/matcher.py        required/preferred/implied tiering, education scoring,
                            eligibility wiring
    utils/scorer.py         recruiter feedback and prioritised recommendations
    utils/parser.py         normalize_job_description() - one cleaning path for all inputs
    utils/sectioning.py     preferred_qualifications section for job descriptions
    utils/extractor.py      dead code removed
    utils/concepts.py       dead code removed
    utils/preprocessing.py  dead code removed
    templates/results.html  Eligibility Screening panel, required/preferred skill rows
    static/css/style.css    additive styles for the new panel only
    tests_eligibility.py    NEW - regression suite
    tests_calibration.py    unused import removed
    README.md               documentation refresh

Unchanged from your original project:

    skills.json
    requirements.txt
    utils/semantic.py
    utils/__init__.py
    templates/base.html, index.html, error.html
    static/js/main.js

Note: uploads/ has been emptied of the sample PDFs. The folder is kept with a
.gitkeep so the app can write to it on first run.
