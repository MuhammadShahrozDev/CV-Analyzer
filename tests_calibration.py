"""Calibration and regression suite.

Six resumes spanning excellent to irrelevant are scored against the same job
description, plus structural edge cases. The point is not that any single
number is right, but that the ordering is correct and the spread is sensible.
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from utils.extractor import extract_skills_from_text, load_skills_config
from utils.matcher import build_match_report
from utils.preprocessing import preprocess_text
from utils.scorer import build_ats_report

cfg = load_skills_config(BASE / "skills.json")

JD = """Data Analyst

We are looking for a Data Analyst to join our analytics team.

Responsibilities:
- Develop dashboards for business stakeholders
- Analyse large datasets and produce actionable insights
- Write and optimise SQL queries
- Build automated reports for leadership
- Partner with cross functional teams

Requirements:
- Strong analytical skills
- Proficiency in Python and SQL
- Experience with a BI tool such as Power BI or Tableau
- Excellent communication skills

Benefits:
- Health insurance, annual bonus, 25 days paid leave, hybrid working.

Equal Opportunity Employer
We value diversity and are an equal opportunity employer.
"""

EXCELLENT = """Ayesha Khan
Senior Data Analyst

PROFESSIONAL SUMMARY
Data analyst with 5 years turning operational data into executive reporting,
specialising in dashboards and self-service analytics for business stakeholders.

TECHNICAL SKILLS
Python, SQL, Power BI, Tableau, Excel, Pandas, NumPy, MySQL, Git, Airflow

WORK EXPERIENCE
Senior Data Analyst, Systems Ltd (2020 - Present)
- Developed interactive Power BI dashboards for business stakeholders across 12 regions
- Analysed large datasets and produced actionable insights that lifted retention 8%
- Wrote and optimised complex SQL queries, cutting report runtime from 40s to 3s
- Built automated reports for leadership, replacing a manual weekly process
- Partnered with cross functional teams in marketing and finance
- Presented findings to senior stakeholders monthly

PROJECTS
Sales Dashboard using Power BI - reporting solution for a retail client
Churn Prediction - scikit-learn model, 84% accuracy

EDUCATION
BS Data Science, UMT, 2019
"""

GOOD = """Ayesha Khan
Data Analyst

PROFESSIONAL SUMMARY
Data analyst with 3 years of experience turning raw operational data into
executive reporting.

TECHNICAL SKILLS
Python, SQL, Power BI, Excel, Pandas, NumPy, MySQL, Git

WORK EXPERIENCE
Data Analyst, Systems Ltd (2022 - Present)
- Built interactive Power BI dashboards tracking sales performance across 12 regions
- Performed data analysis using Python and Pandas to identify revenue leakage
- Wrote complex SQL queries with joins and window functions against MySQL
- Automated weekly reporting, cutting manual effort by 60%
- Presented findings to senior stakeholders every month

PROJECTS
Sales Dashboard using Power BI - end to end reporting solution for a retail client
Customer Churn Prediction - scikit-learn model with 84% accuracy

EDUCATION
BS Data Science, UMT, 2022
"""

PARTIAL = """Usman Tariq
Business Analyst

SUMMARY
Business analyst working with reporting and stakeholder requirements.

SKILLS
Excel, PowerPoint, SQL basics, requirements gathering

WORK EXPERIENCE
Business Analyst, Retail Co (2021 - Present)
- Produced weekly Excel reports for management
- Gathered business requirements from stakeholders
- Ran basic SQL queries against the sales database
- Coordinated with cross functional teams on process changes

EDUCATION
BBA, 2021
"""

JUNIOR = """Sara Malik
Data Science Graduate

SUMMARY
Recent data science graduate looking for an analyst role.

SKILLS
Python, Pandas, SQL, Matplotlib

PROJECTS
Student Performance Analysis - cleaned and analysed a dataset in Pandas
COVID Dashboard - built a Streamlit dashboard showing case trends

EDUCATION
BS Data Science, 2024
"""

IRRELEVANT = """Bilal Ahmed
Graphic Designer

SUMMARY
Creative graphic designer focused on brand identity and print media.

SKILLS
Adobe Photoshop, Illustrator, InDesign, Figma, Branding

WORK EXPERIENCE
Graphic Designer, Studio X (2021 - Present)
- Designed brand identities for 20+ clients
- Produced print collateral and social media assets

EDUCATION
BFA Visual Communication Design, 2021
"""

KEYWORD_STUFFED = """Hamza Ali
Data Analyst

SKILLS
Python, SQL, Power BI, Tableau, dashboards, data analysis, reporting,
communication, teamwork, leadership, analytical skills, business intelligence

WORK EXPERIENCE
Intern, Small Co (2023)
- Helped the team

EDUCATION
BS, 2023
"""


def score(resume, job=JD):
    r = extract_skills_from_text(resume, cfg)
    j = extract_skills_from_text(job, cfg)
    md = build_match_report(
        resume_text=resume, job_text=job,
        resume_clean=preprocess_text(resume), job_clean=preprocess_text(job),
        resume_skills=r, job_skills=j, skills_config=cfg,
        resume_layout_text=resume, job_layout_text=job,
    )
    return build_ats_report(md)


CASES = [
    ("Excellent (tailored)", EXCELLENT, (72, 92)),
    ("Good (relevant)", GOOD, (60, 82)),
    ("Partial (adjacent)", PARTIAL, (30, 60)),
    ("Junior (projects only)", JUNIOR, (25, 62)),
    ("Keyword stuffed (no evidence)", KEYWORD_STUFFED, (30, 65)),
    ("Irrelevant", IRRELEVANT, (0, 20)),
]

if __name__ == "__main__":
    print(f"{'CASE':32} {'ATS':>7} {'skill':>7} {'req':>7} {'kw':>7} {'sim':>7}  EXPECTED  OK")
    print("-" * 92)
    results = []
    for name, resume, (lo, hi) in CASES:
        rep = score(resume)
        c = rep["score_components"]
        ok = lo <= rep["ats_score"] <= hi
        results.append((name, rep["ats_score"], ok))
        print(f"{name:32} {rep['ats_score']:7} {c['skill']:7} {c['requirement']:7} "
              f"{c['keyword']:7} {c['similarity']:7}  {lo:3}-{hi:<3}  {'OK' if ok else 'FAIL'}")

    print()
    by_name = {r[0]: r[1] for r in results}
    ordered = (by_name["Excellent (tailored)"] >= by_name["Good (relevant)"]
               >= by_name["Partial (adjacent)"] >= by_name["Irrelevant"])
    print("Ordering excellent >= good >= partial >= irrelevant:", "OK" if ordered else "FAIL")
    stuffing_ok = by_name["Keyword stuffed (no evidence)"] < by_name["Good (relevant)"]
    print("Keyword stuffing scores below genuine resume:", "OK" if stuffing_ok else "FAIL")
    print("All in expected band:", "OK" if all(r[2] for r in results) else "FAIL")

    print("\n--- EDGE CASES ---")
    for label, resume, job in [
        ("empty job text", GOOD, ""),
        ("empty resume text", "", JD),
        ("no headings at all", "Python SQL Power BI dashboards reporting", JD),
        ("job with no requirements", GOOD, "We are hiring."),
    ]:
        try:
            rep = score(resume, job) if job else score(resume, "n/a")
            print(f"  {label:26} -> ATS {rep['ats_score']} (no crash)")
        except Exception as exc:
            print(f"  {label:26} -> CRASH: {type(exc).__name__}: {exc}")
