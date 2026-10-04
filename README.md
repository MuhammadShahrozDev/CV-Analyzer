# ATS Resume Analyzer - Phase 1

A modular FastAPI application that compares a candidate resume against a job description and produces an ATS compatibility report.

## Features

- Accepts resume uploads in PDF and DOCX formats
- Accepts job descriptions in PDF, DOCX, TXT, or manual text input
- Accepts job descriptions from a public URL, file upload, TXT, or manual text input
- Uses spaCy-based preprocessing with a fallback language pipeline
- Extracts configurable skills from `skills.json`
- Calculates:
  - Keyword match score
  - Skill match score
  - TF-IDF cosine similarity
  - Final ATS score out of 100
- Separates job-description skills into **required**, **preferred** and **implied**, so a
  missing "nice to have" never costs the same as a missing must-have
- Runs an **eligibility screen** (location, work mode, years of experience, education)
  and reports it separately, never folded into the ATS score
- Matches **related degrees**: BS Data Science satisfies "BS Computer Science or related field"
- Normalises pasted, uploaded and linked job descriptions through one cleaning path, so
  the same posting scores identically whichever way it is supplied
- Displays matched and missing skills, matched and missing keywords, and recommendations
- Keeps the code modular so Phase 2 upgrades can add:
  - Sentence Transformers
  - LLM-based analysis
  - Resume rewriting suggestions
  - Multiple job comparison
  - Improved ATS scoring

## Project Structure

```text
resume-analyzer/
├── app.py
├── requirements.txt
├── skills.json
├── uploads/
├── static/
│   ├── css/
│   ├── js/
├── templates/
├── tests_calibration.py
├── tests_eligibility.py
├── uploads/
├── static/
├── templates/
├── utils/
│   ├── parser.py          # file/URL text extraction + JD normalisation
│   ├── preprocessing.py   # spaCy lemmatisation
│   ├── sectioning.py      # resume and job-description section splitting
│   ├── concepts.py        # concept inference and lexical equivalence
│   ├── semantic.py        # embeddings / TF-IDF similarity
│   ├── extractor.py       # skill extraction from skills.json
│   ├── eligibility.py     # location, work mode, experience, education screening
│   ├── matcher.py         # requirement tiering and section analysis
│   └── scorer.py          # ATS score, recruiter feedback, recommendations
└── README.md
```

## Eligibility Screening

Location, work mode, years of experience and education are hiring filters, not
measures of resume quality, so they are screened separately and shown in their
own panel. Mixing them into a match percentage produces a number that means
neither one thing nor the other.

Each check returns Match, Review, Below Requirement or Unknown. Only location,
work mode and education can block a candidate: a shortfall in years routes to a
human rather than auto-rejecting.

## Tests

```bash
python tests_calibration.py   # score ordering and spread across six resumes
python tests_eligibility.py   # tiering, related degrees, screening, input parity
```

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install the spaCy English model for best results:

```bash
python -m spacy download en_core_web_sm
```

3a. Install the Chromium browser Playwright uses as a fallback for
JavaScript-rendered job pages (optional, but recommended - without it, only
statically-renderable job pages can be pulled from a URL):

```bash
playwright install chromium
```

4. Start the app:

```bash
python -m uvicorn app:app --reload --port 8001
```

5. Open the app in your browser:

```text
http://127.0.0.1:8001
```

## Notes

- The app stores uploaded files in the `uploads/` directory.
- The job description input is optional if you paste the job description manually or provide a public URL.
- Manual text takes priority over URL input, and URL input takes priority over the uploaded job file.
- The scoring uses a blended model, not a simple keyword count alone.
- URL extraction pulls only the job posting itself, not the surrounding webpage. It fetches the page, tries `trafilatura` to isolate the main content, then falls back to known job-description containers, and only renders the page in headless Chromium (via Playwright) if the static page turns out to be JavaScript-only. This is what keeps a pasted job description and its URL producing comparable scores.
- Missing Skills always means: catalog skills the job description actually asks for (stated, implied, or near-miss wording) that the resume does not evidence anywhere. Skills from `skills.json` that never appear in the job description at all are never shown as missing.

## Future Upgrade Points

- Replace TF-IDF similarity with Sentence Transformers embeddings
- Add semantic section-aware resume parsing
- Improve skill extraction with synonym and alias maps
- Add LLM-generated rewrite suggestions
- Support comparing one resume against multiple job descriptions
