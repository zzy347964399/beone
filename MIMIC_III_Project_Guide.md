# MIMIC-III Project

---

## What This Is

A hands-on project using any AI coding assistant and any programming language where you explore a real clinical dataset and build something useful — fast. You'll use AI-assisted coding ("vibe coding") to go from raw hospital data to interactive dashboards and statistical models in about 1.5 hours.

We're not testing whether you've memorized survival analysis formulas. We're testing whether you can **direct AI tools effectively**, **ask the right questions of the data**, and **produce clear, working output** under time pressure.

---

## The Dataset: MIMIC-III (Demo)

**MIMIC-III** (Medical Information Mart for Intensive Care) is a freely available clinical database from Beth Israel Deaconess Medical Center. It contains de-identified health records for ~100 patients admitted to ICU, including vitals, labs, medications, diagnoses, and outcomes.

**Download:** [https://www.kaggle.com/datasets/asjad99/mimiciii/data]

Download all CSV files and place them in a single project directory. You should have 26 CSV files including `PATIENTS.csv`, `ADMISSIONS.csv`, `ICUSTAYS.csv`, `LABEVENTS.csv`, `CHARTEVENTS.csv`, `PRESCRIPTIONS.csv`, `DIAGNOSES_ICD.csv`, and others.

### Key facts

| Fact | Detail |
|------|--------|
| Patients | ~100 (demo subset) |
| Admissions | ~129 hospital stays |
| ICU stays | ~136 |
| Lab events | ~76K rows |
| Chart events (vitals) | ~758K rows |
| Prescriptions | ~10K rows |
| Diagnoses | ~1.7K ICD-9 coded entries |
| Time period | Date-shifted (internally consistent, not real calendar dates) |

## Setup (Before You Start)

### 1. Create a Git repository

```bash
mkdir mimic-iii-analysis
cd mimic-iii-analysis
git init
```

Place the downloaded CSV files in this directory. Add a `.gitignore` to exclude large data files:

```
# .gitignore
*.csv
__pycache__/
*.pyc
```


---

## The Project: Three Levels

Work through these in order. Each level builds on the last. It's fine if you don't finish everything — we'd rather see Level 1 and 2 done well than all three done sloppily.

---

### Level 1 — Exploration & Understanding 

**Goal:** Get oriented in the dataset and produce your first visualization.

#### Tasks

**1a. Schema exploration.explore the dataset and visualize the following in a dashboard:
- How many patients died in the hospital? What's the overall mortality rate?
- What are the top 10 most common diagnoses?
- What's the average ICU length of stay? Does it differ by care unit (MICU, SICU, CCU, etc.)?
- Mortality rate by admission type (emergency vs. elective vs. urgent)
- Lab value distributions (pick 3-4 interesting labs)
- ICU length of stay distribution with a median line

**1b. Build an Entity Relationship Diagram.** Create a visualization (any format — HTML, image, text diagram) showing how the key tables connect. Focus on the core tables (PATIENTS, ADMISSIONS, ICUSTAYS, Lab Events) and their relationships to the clinical event tables.



### Level 2 — Analysis & Modeling 

**Goal:** Build a statistical analysis or predictive model and present results in an interactive HTML dashboard.

#### Build models for the following tracks; also, besides following the instructions, can you build better models by trying different approaches and/or features? Also, can you provide relevant summary metrics and explanations of the modeling results?

---

**Track A: Mortality Prediction**

Build a logistic regression model predicting in-hospital mortality (`hospital_expire_flag` in ADMISSIONS).

Suggested features:
- Age, gender, admission type
- Charlson comorbidity index (computed from ICD-9 codes in DIAGNOSES_ICD)
- First-24-hour lab values: extract lab results from LABEVENTS that fall within 24 hours of admission. Key labs: creatinine (itemid 50912), BUN (51006), WBC (51301), platelets (51265), hemoglobin (51222), bilirubin (50885), lactate (50813). Summarize as min/max/mean.


---

**Track B: ICU Length of Stay Prediction**

Build a gradient boosted model predicting ICU length of stay (`los` in ICUSTAYS).

Suggested features:
- Demographics + admission type
- First-24-hour vitals from CHARTEVENTS: heart rate (itemids 211, 220045), systolic BP (51, 220050), diastolic BP (8368, 220051), SpO2 (646, 220277), temperature (223761, 678), respiratory rate (618, 220210). Summarize as min/max/mean.
- Diagnosis count and Charlson index
---


---

### Level 3 — Open-Ended Challenges

**Goal:** Go deeper. These are intentionally open-ended — there's no single right answer.

#### Task: Pharmacovigilance Signal Detection using lab data

Build an eDISH (evaluation of Drug-Induced Serious Hepatotoxicity) scatter plot:
- X-axis: peak ALT / upper limit of normal (ULN = 40 U/L)
- Y-axis: peak total bilirubin / ULN (ULN = 1.2 mg/dL)
- and highlight patients who follow Hy's Law

**Stretch:** Add renal safety (KDIGO staging from creatinine) or hematological toxicity (CTCAE grading for WBC/hemoglobin/platelets).

additional ask: can you connect the different lab data in a visualization or analysis to tell a story of the patients? (any interesting observations, patterns, outliers, or others)
---

## Final Submission

### Push your repository

Look forward to your demo!


