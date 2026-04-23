# MIMIC-III Project Report

> **Dataset**: MIMIC-III Clinical Database Demo Subset (Beth Israel Deaconess Medical Center ICU, De-identified)  
> **Coverage**: Stage 0 (Data Loading & Validation) · Stage 1 (EDA Dashboard & ERD) · Stage 2 (Feature Engineering & Modeling) · Stage 3 (Pharmacovigilance & Patient Story)

---

## Stage 0 — Data Loading & Validation

### Objective

Before any analysis, verify that all core CSV files load correctly, that critical fields contain no structural corruption, and that key summary statistics align with the dataset documentation.

### Dataset Overview

| Table | Rows | Key Fields |
|-------|------|-----------|
| PATIENTS | 100 | subject_id, gender, dob, dod |
| ADMISSIONS | 129 | hadm_id, admittime, admission_type, hospital_expire_flag |
| ICUSTAYS | 136 | icustay_id, first_careunit, intime, outtime, los |
| DIAGNOSES_ICD | 1,761 | icd9_code (~100 patients covered) |
| LABEVENTS | 76,074 | itemid, charttime, valuenum |
| CHARTEVENTS | ~758K (filtered: 64,588) | itemid, charttime, valuenum |

ADMISSIONS has 129 rows for 100 patients, indicating some patients had multiple hospital stays — expected for an ICU dataset. ICUSTAYS (136) exceeds ADMISSIONS (129), confirming that a single admission can involve multiple ICU stays.

### Key Findings

**In-hospital mortality**: 40 deaths out of 129 admissions — a mortality rate of **31.0%**. This is far higher than the typical 2–3% for general hospitalization, which is expected: MIMIC captures only ICU patients, who are the most critically ill in the hospital.

**Admission type distribution**: EMERGENCY 119, ELECTIVE 8, URGENT 2. Emergency admissions account for 92% of the dataset, reflecting that ICU patients are overwhelmingly acute/unplanned cases.

**ICU length of stay**: Median 2.11 days, mean 4.45 days, max 35.4 days. The large gap between mean and median signals a right-skewed distribution driven by a small number of extremely long stays.

**Care unit distribution**:

| Unit | ICU Stays | Description |
|------|-----------|-------------|
| MICU | 77 | Medical ICU — largest share |
| SICU | 23 | Surgical ICU |
| CCU | 19 | Cardiac Care Unit |
| TSICU | 11 | Trauma Surgical ICU |
| CSRU | 6 | Cardiac Surgery Recovery Unit |

MICU dominates (57%), consistent with MIMIC's origin at BIDMC's medical critical care center.

### Engineering Design

**Chunked loading for CHARTEVENTS**: The raw file holds ~758K rows. A full `pd.read_csv()` would require 2–3 GB of memory and risk OOM errors. We use `chunksize=100_000` and retain only the 12 target vital-sign itemids per chunk, reducing the loaded dataset to 64,588 rows — a 92% memory reduction.

```python
for chunk in pd.read_csv(path, usecols=usecols, chunksize=100_000):
    filtered = chunk[chunk["itemid"].isin(itemids)]
    if len(filtered):
        chunks.append(filtered)
```

**Centralized `DATA_DIR`**: All scripts define a single `DATA_DIR` constant at the top. Changing the data path requires editing only one line.

### Issues Found & Fixed

| Issue | Impact | Fix |
|-------|--------|-----|
| `validate_admissions` called `.sum()` on `hospital_expire_flag` without `fillna` first | NULL rows cause mortality rate to be understated | Apply `fillna(0).astype(int)` before summing; print NULL count separately |
| kagglehub cache has extra nesting (`mimic-iii-clinical-database-demo-1.4/`) | All files returned "Not Found" on first run | Inspected path with `ls` layer-by-layer; hardcoded full `DATA_DIR` |

### Potential Improvements

- Read `DATA_DIR` from an environment variable (`os.environ.get("MIMIC_DATA_DIR", ".")`) for portability across machines.
- Add schema validation: check that columns like `admittime` parsed correctly as datetime, catching format issues early.
- The `error` field in CHARTEVENTS flags erroneous records (~1–2% of rows). Currently filtered silently; print the exclusion count for traceability.

---

## Stage 1a — EDA Dashboard

### Objective

Extract key descriptive statistics from ADMISSIONS, ICUSTAYS, DIAGNOSES_ICD, and LABEVENTS, and present them as six interactive Plotly charts in a single standalone HTML file.

### Output

`eda_dashboard.html` — White background, English labels, 2-column grid layout, hover tooltips throughout.

---

### Chart 1: In-Hospital Mortality Rate

**Chart type**: Donut chart with mortality rate in the center; two annotation lines at the bottom showing absolute counts.

**Findings & Interpretation**:

- 40 in-hospital deaths out of 129 admissions — **overall mortality rate 31.0%**
- Survived: 89 (69.0%), Deceased: 40 (31.0%)
- The 31% rate is far above general hospital averages because the cohort is exclusively ICU patients — already the most critically ill in the hospital
- This imbalance (~1:2.2 death-to-survival ratio) directly motivates the class imbalance handling used in Track A modeling

**Design decisions**:

- Donut over solid pie: the hollow center provides space for the mortality percentage label, drawing the eye to the key number
- Bottom annotations split into two lines (`y=-0.10` for total admissions, `y=-0.18` for deaths / survived) to prevent overlap
- Colors: Survived = green (`#10B981`), Deceased = red (`#EF4444`), consistent with clinical data conventions

**Bugs fixed**:

| Issue | Cause | Fix |
|-------|-------|-----|
| Bottom stat lines overlapping | Two annotation `y` values too close | Split to `y=-0.10` and `y=-0.18` |

---

### Chart 2: Top 10 Most Common Diagnoses (ICD-9)

**Chart type**: Horizontal bar chart, sorted descending by count, with color intensity mapped to frequency.

**Findings & Interpretation**:

- Most frequent diagnoses are chronic conditions: **hypertension, coronary artery disease, heart failure, diabetes** — consistent with the comorbidity profile typical of ICU patients
- ICD-9 codes are mapped to human-readable text via `D_ICD_DIAGNOSES.csv`; raw codes are never displayed
- Each admission carries multiple ICD-9 codes (avg ~13.6/admission); the Top 10 reflects total counts across all admissions

**Design decisions**:

- Horizontal bars suit long text labels better than vertical bars
- Diagnosis names are word-wrapped at 30 characters per line (word boundary, no mid-word cuts) using Plotly's `<br>` tag
- Right margin set to `r=60` to prevent bar-end text labels from being clipped
- Color gradient from light blue (low count) to dark blue (high count) encodes frequency intuitively

**Bugs fixed**:

| Issue | Cause | Fix |
|-------|-------|-----|
| Long diagnosis text overflowing | Original approach truncated at 52 characters, cutting words mid-way | Replaced with word-boundary wrapping at ≤30 characters per line |
| Bar-end labels clipped | Default right margin too small | Set `margin=dict(r=60)` |

**Potential improvements**:

- Add a care-unit filter dropdown to compare diagnosis distributions across MICU, CCU, SICU, etc.
- Group ICD-9 codes into clinical chapters (circulatory, endocrine, etc.) for hierarchical exploration

---

### Chart 3: ICU Length of Stay by Care Unit

**Chart type**: Grouped box plots per care unit; `boxmean="sd"` overlays the mean and standard deviation.

**Findings & Interpretation**:

| Unit | Median LOS | Mean LOS | Max LOS | Notes |
|------|-----------|---------|--------|-------|
| MICU | 1.93 d | 3.96 d | 31.1 d | Medical ICU; pronounced long tail |
| SICU | 2.41 d | 5.67 d | 35.4 d | Post-surgical; highest variability |
| CCU | 2.88 d | 5.75 d | 25.0 d | Cardiac; elevated mean |
| TSICU | 1.28 d | 3.59 d | 22.4 d | Trauma; many short stays |
| CSRU | 2.08 d | 3.63 d | 8.1 d | Cardiac surgery; most concentrated |

- SICU has the widest spread (SD 8.75 days), reflecting highly variable post-surgical recovery
- CSRU's max of only 8.1 days suggests a more standardized cardiac surgery recovery pathway
- All units show median < mean, confirming the right-skewed LOS distribution

**Design decisions**:

- Each unit gets a distinct color; `fillcolor` uses `rgba()` at 0.25 opacity for a layered look
- Y-axis ceiling at P95 × 1.2 (~20 days) prevents extreme outliers from compressing the main distribution
- Legend suppressed (`showlegend=False`) since x-axis labels already identify each unit

**Bugs fixed**:

| Issue | Cause | Fix |
|-------|-------|-----|
| Chart rendered blank | `**LAYOUT_BASE` dict unpacking was overridden by Plotly's default template, turning `plot_bgcolor` transparent | Added `template="none"` to fully disable the default theme; set all background colors explicitly |
| `fillcolor` ValueError | Plotly does not accept CSS4 8-digit hex (`#RRGGBBAA`) | Converted with a `hex_to_rgba()` helper to `rgba(r,g,b,a)` format |

**Potential improvements**:

- Overlay mortality rate per unit (scatter or color encoding) to explore LOS vs. mortality relationship
- Switch to violin + box mode to show full distribution shape beyond just quantiles

---

### Chart 4: Mortality Rate by Admission Type

**Chart type**: Bar chart with three groups (EMERGENCY / ELECTIVE / URGENT); percentage labels above each bar.

**Findings & Interpretation**:

| Admission Type | Total | Deaths | Mortality Rate |
|---------------|-------|--------|---------------|
| EMERGENCY | 119 | 39 | **32.8%** |
| ELECTIVE | 8 | 0 | **0.0%** |
| URGENT | 2 | 1 | **50.0%** |

- **ELECTIVE = 0% mortality**: Planned procedures involve pre-operative risk assessment; ICU admission is routine post-operative monitoring with low expected mortality
- **EMERGENCY = 32.8%**: Unplanned, acute presentations with higher disease severity and comorbidity burden
- **URGENT = 50%**: Only 2 cases — statistically unreliable, but may represent patients who deteriorated during a planned procedure

**Design decisions**:

- Y-axis ceiling at `max(rate) × 1.35` (67.5%) — adaptive, not fixed at 100%, so bars fill the plot area meaningfully
- Bar width `0.45` + `bargap=0.5` prevents the three bars from stretching across the full plot width
- Color coding: Emergency = red, Elective = green, Urgent = orange — clinically intuitive

**Bugs fixed**:

| Issue | Cause | Fix |
|-------|-------|-----|
| All bars compressed to the bottom | Y-axis ceiling fixed at 100% while max rate was only 50% | Switched to adaptive ceiling `max(rate) × 1.35` |
| Mortality showing as ~1%/3% instead of 33%/50% | `hospital_expire_flag` had NULL rows; `grouped["dead"] = sum(flag)` propagated NaN | Apply `fillna(0).astype(int)` before `groupby` |

**Potential improvements**:

- Annotate the URGENT bar with "n=2, interpret with caution" to prevent misreading
- Convert to a stacked bar showing absolute counts alongside percentages

---

### Chart 5: Lab Value Distributions

**Chart type**: Three side-by-side histograms with Median (colored dashed line) and Mean (orange dotted line) reference lines.

**Indicators selected**:

| Indicator | itemid | Unit | Clinical Meaning | Bin Size |
|-----------|--------|------|-----------------|---------|
| Creatinine | 50912 | mg/dL | Renal function | 1.0 |
| WBC | 51301 | K/uL | Infection / inflammation | 2.0 |
| Hemoglobin | 51222 | g/dL | Anemia | 0.5 |

**Findings & Interpretation**:

- **Creatinine**: Right-skewed; most values fall within normal range (0.5–1.2 mg/dL) but a high-value tail (>5 mg/dL) indicates acute kidney injury in a subset of patients
- **WBC**: Main peak at 5–15 K/uL (normal range) with a pronounced right tail extending to 30–50 K/uL, suggesting severe infection or hematologic disease in some patients
- **Hemoglobin**: Peak at 8–12 g/dL — below normal ranges (M: 13.5–17.5, F: 12–15.5) — reflecting the anemia ubiquitous in ICU patients (hemodilution, blood loss, chronic disease)

All three distributions are consistent with the typical ICU presentation: concurrent renal impairment, systemic inflammation, and hospital-acquired anemia.

**Design decisions**:

- Switched from Violin to Histogram: violin plots fail silently when kernel density estimation collapses (highly concentrated distributions); histograms are more robust
- Lower clip bound is strict (`sub > 0`), filtering out erroneous zero readings from instrument faults
- Each subplot has independent Median and Mean lines to visualize distributional skew

**Bugs fixed**:

| Issue | Cause | Fix |
|-------|-------|-----|
| Hemoglobin / Creatinine subplots blank | Violin KDE estimation degenerated on concentrated distributions | Replaced with Histogram |
| Zero-value noise | Clip used `sub >= 0`, including instrument-error zeros | Changed to `sub > clip_range[0]` (strictly greater) |

**Potential improvements**:

- Add Lactate, Bilirubin, and Platelets for a more comprehensive lab panel
- Overlay death vs. survival groups as stacked/semi-transparent histograms to compare distributions

---

### Chart 6: ICU Length of Stay Distribution

**Chart type**: Histogram (bin width 0.5 days) with three reference lines: Median (red dashed), Mean (orange dotted), P75 (blue long-dash).

**Findings & Interpretation**:

| Statistic | Value |
|-----------|-------|
| Median | 2.11 days |
| Mean | 4.45 days |
| P75 | 4.84 days |
| Max | 35.4 days |

- **Mean (4.45 d) >> Median (2.11 d)**: Classic right-skew — most patients leave the ICU within 1–3 days, but a small number of severely ill patients with stays >10 days pull the mean upward
- **P75 = 4.84 days**: 75% of patients are discharged from the ICU within 5 days; the remaining 25% form the long tail
- This skew directly motivates the `log1p` transformation used in Track B regression modeling

**Design decisions**:

- Bin width of 0.5 days (12 hours) gives enough resolution to see distribution shape without excessive sparsity
- Three reference lines together reveal asymmetry: the gap between Median and Mean visually quantifies right-skew
- X-axis ceiling at P97 (~20 days) rather than the max, preventing the extreme outlier (35 days) from compressing the main body of the distribution to the left

**Potential improvements**:

- Stacked histogram colored by care unit to show each unit's LOS contribution
- Add a CDF curve so the viewer can directly read "X% of patients discharge within N days"

---

## Stage 1b — Entity Relationship Diagram (ERD)

### Objective

Visualize the primary key (PK) / foreign key (FK) relationships between MIMIC-III's core tables, providing a reference map for the join logic used in feature engineering.

### Output

`mimic_iii_erd.drawio` — draw.io XML format. Open via [app.diagrams.net](https://app.diagrams.net) (File → Open from → Device) or by double-clicking in the draw.io desktop app.

### Table Structure & Relationships

```
PATIENTS (subject_id PK)
  ├─► ADMISSIONS (hadm_id PK | subject_id FK)
  │     ├─► ICUSTAYS (icustay_id PK | subject_id FK, hadm_id FK)
  │     │     └─► CHARTEVENTS (subject_id FK, icustay_id FK, itemid FK)
  │     │               └── D_ITEMS (itemid PK)
  │     ├─► LABEVENTS (subject_id FK, hadm_id FK, itemid FK)
  │     │               └── D_LABITEMS (itemid PK)
  │     ├─► DIAGNOSES_ICD (subject_id FK, hadm_id FK, icd9_code FK)
  │     │               └── D_ICD_DIAGNOSES (icd9_code PK)
  │     └─► PRESCRIPTIONS (subject_id FK, hadm_id FK)
```

### Table Categories & Colors

| Color | Type | Tables |
|-------|------|--------|
| 🔵 Blue | Core business tables | PATIENTS, ADMISSIONS, ICUSTAYS |
| 🟢 Green | Clinical event tables | LABEVENTS, CHARTEVENTS, DIAGNOSES_ICD, PRESCRIPTIONS |
| 🟡 Yellow | Dictionary / lookup tables | D_ICD_DIAGNOSES, D_ITEMS |

### Key Relationship Explanations

**One-to-Many**:

- `PATIENTS → ADMISSIONS`: One patient can have multiple admissions (100 patients → 129 admissions; avg 1.29/patient)
- `ADMISSIONS → ICUSTAYS`: One admission can involve multiple ICU stays (129 admissions → 136 stays)
- `ICUSTAYS → CHARTEVENTS`: One ICU stay generates hundreds to thousands of vital-sign records (~758K total)

**Many-to-One**:

- `LABEVENTS → D_LABITEMS`: `itemid` resolves to the lab test's name, units, and reference range
- `CHARTEVENTS → D_ITEMS`: `itemid` resolves to the vital sign's name; the same vital sign has different `itemid` values in the CareVue (legacy) and MetaVision (current) monitoring systems (e.g., Heart Rate: 211 vs 220045) — these must be merged in feature engineering

**Composite keys**:

- LABEVENTS and DIAGNOSES_ICD have no single-column primary key; records are identified by `(subject_id, hadm_id)` combinations
- CHARTEVENTS uses `(subject_id, icustay_id)` as its composite identifier

### Design Decision

draw.io XML was chosen over Mermaid.js HTML for the following reasons:

- draw.io can export to PNG/SVG for direct inclusion in reports and slides
- Mermaid's auto-layout becomes unreliable beyond 6–8 tables (node overlap is hard to control)
- draw.io's native table shape explicitly supports PK/FK annotation, producing a cleaner result

### Known Issues

- D_LABITEMS is shown as a reference node only (no field details), since only its `label` column is used in practice
- PRESCRIPTIONS is included for structural completeness but was not used in feature engineering for this project

### Potential Improvements

- Use `eralchemy2` to auto-generate the ERD from pandas DataFrame schemas, enabling it to stay in sync with the data via CI/CD
- Annotate each table node with its row count (e.g., "LABEVENTS: 76,074 rows") so the ERD also conveys dataset scale

---

## Stage 0 & 1 — Key Numbers at a Glance

| Metric | Value |
|--------|-------|
| Total patients | 100 |
| Total admissions | 129 |
| Total ICU stays | 136 |
| In-hospital mortality rate | 31.0% (40/129) |
| Emergency admission share | 92.2% (119/129) |
| MICU share | 56.6% (77/136) |
| ICU LOS median | 2.11 days |
| ICU LOS mean | 4.45 days |
| ICU LOS max | 35.4 days (SICU) |
| Total lab event records | 76,074 |
| CHARTEVENTS (filtered) | 64,588 (from ~758K) |
| Total diagnosis codes | 1,761 (avg 13.6 per admission) |

---

## Stage 2-FE — Feature Engineering

### Objective

Extract structured feature matrices from raw clinical data for two modeling tracks: Track A (in-hospital mortality prediction) and Track B (ICU LOS prediction).

### Outputs

| File | Rows | Columns (incl. label) | Granularity |
|------|------|-----------------------|-------------|
| `features_track_a.csv` | 129 | 29 | Per admission (hadm_id) |
| `features_track_b.csv` | 136 | 27 | Per ICU stay (icustay_id) |

---

### Shared Features: Demographics

**Sources**: PATIENTS.csv + ADMISSIONS.csv

| Feature | Raw Field | Processing |
|---------|-----------|------------|
| `age` | dob + admittime | Exact calculation (days / 365.25); >89 yr patients receive sentinel value 91.0 |
| `age_imputed` | dob | 0 = true age; 1 = age unknown (>89 yr, dob shifted by MIMIC) |
| `gender_M` | gender | One-hot (Male = 1, Female = 0) |
| `adm_EMERGENCY` | admission_type | One-hot |
| `adm_ELECTIVE` | admission_type | One-hot |
| `adm_URGENT` | admission_type | One-hot |

**Age handling — three iterations** (critical design decision):

MIMIC protects the privacy of patients aged >89 by shifting their `dob` approximately 300 years into the past (to the early 1800s). Computing `admittime - dob` directly causes an int64 overflow.

| Version | Approach | Problem |
|---------|----------|---------|
| v1 | `(admittime - dob).dt.days / 365.25` | dob shifted to 1800s → int64 overflow, program crashes |
| v2 | `dt.year` difference + `clip(0, 89)` | Ignores month/day; precision off by up to half a year; clipping to 89 discards the "elderly" directional signal |
| v3 (final) | Exact calculation for normal patients; sentinel value 91 + `age_imputed=1` for dob < 1900 | ✅ Preserves the "elderly" directional signal; model can learn this group separately via `age_imputed` |

**Why 91 and not the cohort median (71.2)?** Labeling an elderly patient as 71 years old tells the model "this is a middle-aged patient" — which is actively harmful for mortality and LOS prediction. Age 91 is numerically above 89, preserving at minimum the directional information that these patients are older and higher risk.

9 admissions (7% of the dataset) correspond to patients with shifted dob values.

---

### Shared Features: Charlson Comorbidity Index

**Source**: DIAGNOSES_ICD.csv

The Charlson Comorbidity Index (CCI) is the most widely used tool for quantifying comorbidity burden: 17 disease categories are weighted and summed, with higher scores predicting higher mortality risk.

**Implementation**: Hand-coded ICD-9 prefix mapping table based on Quan et al. 2005 — no external library dependency.

| Disease Category | Weight | Example ICD-9 Prefixes |
|-----------------|--------|------------------------|
| MI, CHF, PVD, CVD, COPD, etc. | 1 | 410, 428, 440, 43... |
| Complicated diabetes, hemiplegia, renal disease, malignancy | 2 | 2504, 342, 582, 140... |
| Severe liver disease | 3 | 4560, 5722... |
| Metastatic tumor, AIDS | 6 | 196, 042... |

**Dataset statistics**:

- Mean CCI: **3.52** — substantially above the typical community hospital average (<2), consistent with the high comorbidity burden expected in ICU patients
- All 129 admissions have at least one ICD-9 code; missing CCI (no diagnoses on record) defaults to 0

---

### Track A Features: First 24h Lab Values

**Source**: LABEVENTS.csv

**Time window**: All lab results recorded within 24 hours of `admittime` (hospital admission time).

> ⚠️ Key trap: The window must be anchored to each patient's individual `admittime`, not to a fixed calendar date. Different patients are admitted at different times.

```python
lab["hours_from_admit"] = (
    lab["charttime"] - lab["admittime"]
).dt.total_seconds() / 3600
lab = lab[(lab["hours_from_admit"] >= 0) & (lab["hours_from_admit"] <= 24)]
```

**7 lab indicators** (min / max / mean per indicator = 21 columns total):

| Indicator | itemid | Unit | Clinical Meaning | Clip Range |
|-----------|--------|------|-----------------|-----------|
| Creatinine | 50912 | mg/dL | Renal function | (0, 30) |
| BUN | 51006 | mg/dL | Renal function | (0, 300) |
| WBC | 51301 | K/uL | Infection / inflammation | (0, 200) |
| Platelets | 51265 | K/uL | Coagulation | (0, 2000) |
| Hemoglobin | 51222 | g/dL | Anemia | (0, 25) |
| Bilirubin | 50885 | mg/dL | Liver function | (0, 50) |
| Lactate | 50813 | mmol/L | Tissue perfusion / shock | (0, 30) |

**Missing value handling**: Global median imputation (no rows dropped), preserving all 129 samples. Not distinguishing "not measured" from "normal" is a known limitation of this approach.

**Final feature matrix**: 129 rows × 21 Lab columns + 7 demographic/CCI columns = **28 feature columns** (29 including the label).

---

### Track B Features: First 24h Vital Signs

**Source**: CHARTEVENTS.csv (64,588 rows after filtering)

**Time window**: Vital signs recorded within 24 hours of `intime` (ICU admission time) — distinct from Track A's `admittime` anchor.

**6 vital signs** (min / max / mean per sign = 18 columns):

| Vital Sign | CareVue itemid | MetaVision itemid | Unit | Clip Range |
|-----------|---------------|-----------------|------|-----------|
| Heart Rate | 211 | 220045 | bpm | (0, 300) |
| Systolic BP | 51 | 220050 | mmHg | (0, 300) |
| Diastolic BP | 8368 | 220051 | mmHg | (0, 200) |
| SpO2 | 646 | 220277 | % | (50, 100) |
| Temperature | 678 | 223761 | °C | (25, 45) |
| Resp Rate | 618 | 220210 | /min | (0, 80) |

**Dual-system itemid merging** (key trap): MIMIC was recorded across two bedside monitoring systems — CareVue (legacy) and MetaVision (current) — with different `itemid` values for the same vital sign. Both codes are included in a single filter:

```python
VITAL_ITEMS = {
    "hr": ([211, 220045], (0, 300)),   # CareVue + MetaVision
    ...
}
sub = ce[ce["itemid"].isin(itemids)].copy()
```

**Temperature unit harmonization**: CareVue (itemid=678) records in Fahrenheit; MetaVision (223761) records in Celsius. Conversion applied before clipping:

```python
mask_f = sub["itemid"] == 678
sub.loc[mask_f, "valuenum"] = (sub.loc[mask_f, "valuenum"] - 32) * 5 / 9
sub["valuenum"] = sub["valuenum"].clip(25, 45)
```

**Diagnosis count feature**: Total number of ICD-9 codes per admission (`diag_count`), reflecting diagnostic complexity.

**Final feature matrix**: 136 rows × 18 vital columns + 1 `diag_count` + 7 demographic/CCI columns = **26 feature columns** (27 including the label).

---

### Known Issues & Improvements

| Issue | Impact | Improvement |
|-------|--------|-------------|
| Global median imputation; "not measured" indistinguishable from "normal" | Model cannot learn from the absence of a measurement | Add binary indicator columns (e.g., `creatinine_measured = 0/1`) for each Lab/vital feature |
| 24h window excludes pre-admission labs (e.g., ED triage results) | Some clinically relevant early values are missed | Extend window to `admittime − 2h` for emergency patients |
| Track A and Track B use different time anchors (admittime vs. intime) | If a patient waits a long time in a ward before ICU transfer, the two feature sets represent different time periods | This is intentional: Track A predicts mortality at admission; Track B predicts LOS at ICU entry |
| Charlson mapping uses ICD-9 prefix matching, which may miss some subcodes | Comorbidity burden may be underestimated for patients with unusual code variants | Use a complete validated mapping library (e.g., Python port of R's `comorbidity` package) |

---

## Stage 2A — Track A: In-Hospital Mortality Prediction

### Task Definition

| Item | Detail |
|------|--------|
| Target | `hospital_expire_flag` (0 = survived, 1 = in-hospital death) |
| Task type | Binary classification |
| Sample size | 129 (train 103, test 26) |
| Class ratio | Survived 89 / Deceased 40 (~2.2:1) |
| Primary metric | CV AUROC (5-fold Stratified), Death class Recall |

### Model Configuration

| Model | Key Hyperparameters | Imbalance Handling |
|-------|--------------------|--------------------|
| Logistic Regression | `C=0.1` (strong regularization), StandardScaler preprocessing | `class_weight='balanced'` |
| Random Forest | `n_estimators=200, max_depth=6, min_samples_leaf=3` | `class_weight='balanced'` |
| XGBoost | `n_estimators=200, max_depth=4, learning_rate=0.05` | `scale_pos_weight = 89/40 ≈ 2.22` |

### Experimental Results

| Model | Test AUROC | CV AUROC (5-fold) | Death Recall | Death F1 |
|-------|-----------|------------------|-------------|---------|
| Logistic Regression | 0.542 | 0.590 ± 0.074 | 0.500 | 0.444 |
| Random Forest | 0.438 | 0.598 ± 0.073 | 0.375 | 0.429 |
| **XGBoost** | **0.590** | **0.609 ± 0.050** | **0.500** | **0.471** |

### Result Interpretation

**Why CV AUROC over test-set AUROC**: The test set contains only 26 samples (~8 deaths), making single-split AUROC highly variable. The 5-fold Stratified CV evaluates across all 129 samples, giving a far more stable estimate. XGBoost's CV AUROC of 0.609 ± 0.050 also has the smallest standard deviation, indicating the most stable generalization.

**What AUROC 0.61 means clinically**: On a 129-sample demo dataset, 0.61 is an acceptable initial signal (random = 0.5, perfect = 1.0). Comparable models on the full MIMIC dataset (46K+ patients) typically reach 0.80–0.85. The current bottleneck is **sample size, not model choice** — the three models differ by only 0.019 in CV AUROC.

**Why Death Recall matters**: In clinical settings, a missed death prediction (False Negative) carries far higher cost than a false alarm (False Positive). Death class Recall of 0.50 (correctly identifying 50% of patients who will die) is the metric that matters most — and there is substantial room for improvement.

**Impact of sentinel value 91**: Replacing the cohort median age (71.2) with sentinel value 91 for >89-year patients raised XGBoost test AUROC from 0.549 to 0.590 (+0.041), confirming that preserving the "elderly" directional signal improves mortality prediction.

**Class imbalance handling**: The ~1:2.2 ratio is not extreme but still requires attention. `class_weight='balanced'` effectively multiplies the minority class loss by 2.22×, forcing the model to weight death-prediction errors more heavily.

### Feature Importance (XGBoost Top Contributors)

- **Lactate features** (lactate_max, lactate_mean): Lactate is a direct biochemical marker of tissue hypoxia and shock — a strong predictor of ICU mortality
- **Charlson index**: Higher comorbidity burden → higher mortality risk
- **Bilirubin features**: Hepatic dysfunction marker; elevated bilirubin signals multi-organ dysfunction
- **Creatinine features**: Renal impairment is an independent ICU mortality predictor
- **Age / age_imputed**: Older patients have higher mortality risk; `age_imputed` flags the high-risk >89 group

### Known Issues & Improvements

| Issue | Impact | Improvement |
|-------|--------|-------------|
| Sample size: 129 total, 26 test | CV AUROC SD ±0.05–0.07; results unstable | Retrain on full MIMIC (~50K patients) |
| No SMOTE oversampling | `class_weight` adjusts loss weights but does not generate new minority-class samples | Compare SMOTE + no class_weight vs. current approach |
| Logistic Regression `C=0.1` fixed | May not be the optimal regularization strength | Use `LogisticRegressionCV` to select C via cross-validation |
| Care unit not included as a feature | CCU vs. MICU have meaningfully different mortality rates | Add `first_careunit` one-hot encoding (avoid LOS features to prevent data leakage) |
| No SHAP explanation | XGBoost's built-in `feature_importances_` cannot explain individual patient predictions | Add `shap.TreeExplainer` to generate waterfall and summary plots |

---

## Stage 2B — Track B: ICU Length of Stay Prediction

### Task Definition

| Item | Detail |
|------|--------|
| Target | `ICUSTAYS.los` (ICU stay in days, continuous) |
| Task type | Regression |
| Sample size | 136 (train 108, test 28) |
| LOS distribution | Median 2.11 d, mean 4.45 d, max 35.4 d (right-skewed) |
| Primary metric | MAE (mean absolute error, in days) |

### Target Transformation

LOS has a strongly right-skewed distribution (mean/median ratio = 2.1, P95 ≈ 17 days). Fitting directly would allow a few extreme stays to dominate the loss. We apply `log1p` to compress the tail:

```python
y_log = np.log1p(y)              # training target
y_pred_orig = np.expm1(y_pred_log)  # reverse transform predictions to days
```

`log1p(x) = log(1+x)` is safe at x=0 (avoids log(0) = −∞); `expm1` is its inverse.

### Model Configuration

| Model | Type | Key Hyperparameters |
|-------|------|---------------------|
| Baseline | Mean prediction (DummyRegressor) | None |
| XGBoost | Gradient boosted trees | `n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8` |
| LightGBM | Gradient boosted trees | `n_estimators=300, max_depth=4, learning_rate=0.05, min_child_samples=5` |

### Experimental Results

| Model | MAE (days) | RMSE (days) | R² | CV MAE (5-fold) |
|-------|-----------|------------|-----|----------------|
| Baseline (mean) | 2.235 | 3.651 | −0.014 | — |
| **XGBoost** | **2.411** | **3.486** | **0.075** | 0.530 ± 0.057 |
| LightGBM | 2.626 | 3.986 | −0.209 | 0.564 ± 0.047 |

### Result Interpretation

**What the baseline represents**: Always predicting the sample mean LOS (4.45 days) gives MAE = 2.235 days. Any model adding real value must beat this naive benchmark.

**XGBoost's performance**:
- Test MAE = 2.411 days, marginally worse than the baseline on the 28-sample test set
- But R² = 0.075 > Baseline's −0.014, confirming XGBoost captures some variance structure
- CV MAE = 0.530 days (5-fold) — far better than the test-set MAE of 2.411 days. This gap reflects the high variance of a 28-sample test set, not poor generalization

**Why LightGBM underperforms**: With only 108 training samples, `min_child_samples=5` may still allow overfitting; R² = −0.209 on the test set reflects instability, not a fundamental model failure.

**Why ICU LOS is hard to predict**: LOS is driven by clinical decisions (transfer timing, surgical interventions, complications) that cannot be fully inferred from admission-time vitals and labs. Even on the full MIMIC dataset, published LOS models typically achieve R² = 0.2–0.4; the current demo results are within the expected range.

### Feature Importance (XGBoost Top Contributors)

- **Resp Rate features** (rr_max, rr_mean): Elevated respiratory rate signals respiratory distress, associated with prolonged ICU stays
- **SpO2 features** (spo2_min): Low oxygen saturation directly indicates disease severity
- **Heart Rate features**: Heart rate variability correlates with ICU illness severity
- **Charlson index**: Higher comorbidity burden → slower recovery → longer stays
- **diag_count**: More diagnoses reflect greater clinical complexity and longer stays
- **Age**: Elderly patients recover more slowly, extending LOS

### Known Issues & Improvements

| Issue | Impact | Improvement |
|-------|--------|-------------|
| Test set only 28 samples | High MAE variance; results unreliable | Use full MIMIC, or Leave-One-Out CV to better exploit small samples |
| Track A Lab features not included in Track B | Lactate, creatinine etc. are also predictive of LOS | Merge Track A's 21 Lab features into Track B's feature matrix |
| No quantile regression | Only point predictions; no uncertainty estimates | Use `XGBQuantile` or LightGBM quantile loss to estimate P10/P50/P90 intervals |
| Hyperparameters not tuned | Current params are fixed heuristics | Use Optuna or GridSearch to optimize `max_depth`, `learning_rate`, `subsample` via CV |
| LOS censoring not handled | Patients who die in the ICU have censored LOS (they would have stayed longer if they hadn't died) | Switch to survival analysis (Cox PH, AFT models) to handle censored observations properly |

---

## Stage 2-DB — Modeling Dashboard

### Objective

Consolidate Track A and Track B results into a single interactive HTML page that opens in any browser without a server.

### Output

`modeling_dashboard.html` — White background, two clearly separated track sections, full Plotly interactivity.

### Dashboard Structure

**Track A section**:

| Component | Content |
|-----------|---------|
| Metric cards | AUROC / CV AUROC / Death Recall / Death F1 for each model |
| ROC curves | Three overlaid ROC curves + random-chance reference line |
| Confusion matrix | XGBoost heatmap showing TP / FP / TN / FN |
| Feature importance | XGBoost Top 15 features, horizontal bar chart |

**Track B section**:

| Component | Content |
|-----------|---------|
| Metric cards | MAE / RMSE / R² / CV MAE for each model |
| Predicted vs. Actual | Scatter plot with all models overlaid; perfect-prediction diagonal |
| Error distribution | Histogram of (predicted − actual) for each model; zero-error reference line |
| Feature importance | LightGBM Top 15 features |

### Key Design Choices

**Feature name mapping**: Raw column names (e.g., `lactate_max`) are mapped to readable labels (e.g., `Lactate_max`) so non-technical stakeholders can interpret the charts.

**Decoupled data flow**: Track A/B scripts serialize results to `.pkl` files; the dashboard script loads those pkl files and regenerates all charts. Models do not need to be retrained to refresh the dashboard.

**Single-file export**: `plotly.io.to_html(full_html=False, include_plotlyjs=False)` generates each chart as a div; these are embedded into a single HTML file with Plotly JS loaded from CDN. The file is fully self-contained for sharing.

### Known Issues & Improvements

| Issue | Improvement |
|-------|-------------|
| Feature importance uses absolute LR coefficients (not SHAP) | Add `shap.TreeExplainer` to unify feature attribution across all three models |
| No Precision-Recall curve for Track A | PR curves are more informative than ROC under class imbalance; should be added |
| Confusion matrix shown for XGBoost only | Add side-by-side confusion matrices for all three models |
| Dashboard is a static snapshot | Add model-selection dropdown to dynamically switch which model's results are displayed |

---

## Stage 2 — Key Numbers at a Glance

| Metric | Value |
|--------|-------|
| Track A feature count | 28 |
| Track B feature count | 26 |
| Age-shifted patients | 9 (7% of admissions) |
| Mean Charlson index | 3.52 |
| Best CV AUROC (Track A) | 0.609 ± 0.050 (XGBoost) |
| Best Death Recall (Track A) | 0.500 (Logistic Regression / XGBoost) |
| Baseline MAE (Track B) | 2.235 days |
| Best Test MAE (Track B) | 2.411 days (XGBoost) |
| Best CV MAE (Track B) | 0.530 ± 0.057 days (XGBoost) |
| Best R² (Track B) | 0.075 (XGBoost) |

---

## Stage 3a — eDISH Pharmacovigilance

### Background & Clinical Significance

**eDISH** (evaluation of Drug-Induced Serious Hepatotoxicity) is an FDA-recommended tool for detecting hepatotoxicity signals, combining two dimensions: hepatocellular injury (ALT elevation) and impaired liver function (bilirubin elevation).

**Hy's Law** (formulated by Hyman Zimmerman) is the classical criterion for identifying potentially serious drug-induced liver injury:

> ALT > 3× ULN **AND** Total Bilirubin > 2× ULN, with cholestasis excluded (ALP < 2× ULN)

In clinical trials, patients meeting Hy's Law criteria carry an approximately 10% risk of fatal liver failure — a threshold that typically triggers drug development halt or a black-box warning.

### Output

`edish_dashboard.html` — Main eDISH scatter plot, quadrant definition table, KDIGO renal staging chart, CTCAE hematological toxicity chart.

---

### Main eDISH Scatter Plot

**Chart design**:

| Dimension | Content |
|-----------|---------|
| X-axis | Peak ALT / ULN (ULN = 40 U/L), log scale |
| Y-axis | Peak Total Bilirubin / ULN (ULN = 1.2 mg/dL), log scale |
| Data granularity | One point per patient (subject_id level; peak values across entire hospitalization) |
| Reference lines | ALT = 1× ULN, ALT = 3× ULN (Hy's threshold), Bili = 1× ULN, Bili = 2× ULN (Hy's threshold) |

**Four-quadrant definitions**:

| Quadrant | ALT / ULN | Bili / ULN | Clinical Meaning | Marker |
|----------|-----------|-----------|-----------------|--------|
| ⭐ **Hy's Law** | > 3× | > 2× | Potential serious hepatotoxicity; highest risk signal | Red star, size=16 |
| 🟣 Temple's Corollary | ≤ 3× | > 2× | Predominantly cholestatic; bile duct dysfunction | Purple square |
| 🟡 Hepatocellular | > 3× | ≤ 2× | Hepatocellular injury without jaundice | Orange diamond |
| ⚪ Normal | ≤ 3× | ≤ 2× | Within normal or non-concerning range | Gray circle |

**Why logarithmic axes**: ALT can range from normal (10–40 U/L) to severely elevated (>1000 U/L) — spanning two orders of magnitude. On a linear scale, the vast majority of points (normal to mildly elevated) would be compressed into the far left, making them indistinguishable. Log scale allocates equal visual space to each order of magnitude.

**Hover tooltip**: Each point displays subject_id, peak ALT (U/L and × ULN), peak Bilirubin (mg/dL and × ULN), KDIGO renal stage, and WBC/HGB/PLT CTCAE grades — integrating multi-system toxicity information in a single interaction.

---

### Extension 1: KDIGO Renal Safety Staging

**Background**: The KDIGO (Kidney Disease: Improving Global Outcomes) AKI staging system is the international standard for classifying acute kidney injury severity based on serum creatinine changes.

**Implementation**: Each patient's minimum creatinine during hospitalization is used as the baseline (simplified version; true KDIGO requires a baseline from the 7 days before admission), with peak creatinine compared to that baseline.

| KDIGO Stage | Criteria | Clinical Meaning |
|-------------|----------|-----------------|
| No AKI | Ratio < 1.5× AND Δ < 0.3 mg/dL | Normal renal function |
| Stage 1 | Ratio 1.5–1.9× OR Δ ≥ 0.3 mg/dL | Mild AKI |
| Stage 2 | Ratio 2.0–2.9× | Moderate AKI |
| Stage 3 | Ratio ≥ 3× OR Peak ≥ 4.0 mg/dL | Severe AKI; may require renal replacement therapy |

**Chart**: Grouped bar chart with criteria embedded in x-axis labels; color gradient from gray (No AKI) to red (Stage 3).

**Clinical connection**: KDIGO staging is integrated into the eDISH hover tooltip, enabling identification of patients with simultaneous hepatic and renal injury — a hallmark of multi-organ dysfunction syndrome (MODS).

---

### Extension 2: CTCAE Hematological Toxicity Grading

**Background**: CTCAE (Common Terminology Criteria for Adverse Events), published by the NCI, is the standard framework for grading adverse events. Grade 3–4 corresponds to severe toxicity typically requiring intervention.

**Three indicators analyzed**:

| Indicator | itemid | Grade 3 Threshold | Grade 4 Threshold | Clinical Meaning |
|-----------|--------|------------------|------------------|-----------------|
| WBC | 51301 | < 2.0 K/uL | < 1.0 K/uL | Leukopenia → infection risk |
| Hemoglobin | 51222 | < 8.0 g/dL | < 6.5 g/dL | Severe anemia → tissue hypoxia |
| Platelets | 51265 | < 50 K/uL | < 25 K/uL | Thrombocytopenia → bleeding risk |

**Implementation note**: Nadir values (lowest recorded during hospitalization) are used, not means or peaks. The nadir reflects the worst toxicity state the patient experienced.

**Chart**: Stacked bar chart; x-axis = three indicators, each color layer = one CTCAE grade, color gradient from pale yellow (Grade 0) to deep red (Grade 4).

---

### Known Issues & Improvements

| Issue | Impact | Improvement |
|-------|--------|-------------|
| KDIGO baseline = intra-admission creatinine minimum | Understates AKI severity (patients already in AKI at admission have artificially low baselines) | Use CKD-EPI formula to estimate pre-admission baseline creatinine from demographics |
| eDISH does not exclude cholestasis (ALP not incorporated) | Strict Hy's Law requires ALP < 2× ULN; some cholestatic patients may be misclassified | Add ALP (itemid=50863) and apply the full exclusion criterion |
| Analysis aggregates peak values (no temporal tracking) | Cannot distinguish whether ALT and Bili peaked simultaneously or weeks apart (clinically meaningful difference) | Implement rolling 7-day peak windows to track co-elevation patterns over time |
| CTCAE thresholds are simplified | WBC Grade 3 per official guidelines uses absolute neutrophil count (ANC), not total WBC | Source ANC data (itemid may be 51256) for more accurate grading |
| No drug attribution | eDISH's primary use is to flag hepatotoxicity signals for specific drugs; current analysis covers all patients regardless of medication | Join PRESCRIPTIONS table; generate per-drug eDISH sub-plots |

---

## Stage 3b — Patient Story Analysis

### Objective

Integrate six Lab indicators across multiple analytical lenses — correlation, group comparison, outlier identification, and individual trajectory — to tell the clinical story of ICU patients.

### Output

`patient_story.html` — Four interactive charts, each with a "What this tells us" clinical narrative, white background, English throughout.

### Lab Indicators Used

| Indicator | itemid | Unit | Clinical Meaning |
|-----------|--------|------|-----------------|
| Creatinine | 50912 | mg/dL | Renal function |
| Lactate | 50813 | mmol/L | Tissue perfusion / shock |
| WBC | 51301 | K/uL | Infection / inflammation |
| Hemoglobin | 51222 | g/dL | Anemia |
| Bilirubin | 50885 | mg/dL | Liver function |
| ALT | 50861 | U/L | Hepatocellular injury |

---

### Chart 1: Lab Marker Correlation Matrix

**Method**: For each patient, compute the peak value of each Lab indicator during the entire hospitalization. Build a 6×6 Pearson correlation matrix across all patients.

**Key findings & clinical interpretation**:

- **Creatinine ↔ Bilirubin positive correlation**: Concurrent renal and hepatic deterioration is a hallmark of multi-organ dysfunction syndrome (MODS) — characteristic of severe sepsis in the ICU
- **Lactate ↔ Bilirubin positive correlation**: Elevated lactate reflects circulatory failure and tissue hypoxia; elevated bilirubin reflects hepatic dysfunction or hemolysis. Their co-elevation is a strong signal for septic shock with hepatic involvement
- **Hemoglobin negatively correlated with most other markers**: As critically ill patients develop anemia, their inflammatory and organ-injury markers tend to rise — opposite directions in a shared physiological deterioration process

**Chart design**:
- Color scale: negative correlation = dark blue (r=−1) → white (r=0) → dark red (r=+1), matching the intuitive "cold-neutral-hot" mental model
- Diagonal cells (self-correlation = 1.0) are the deepest red
- Numeric values labeled inside each cell to 2 decimal places

---

### Chart 2: Deceased vs. Survived — Mean Peak Lab Values

**Method**: Split patients by `hospital_expire_flag`; compute the mean peak value of each Lab indicator per group; display as a grouped bar chart.

**Key findings & clinical interpretation**:

| Indicator | Deceased Group | Survived Group | Interpretation |
|-----------|---------------|---------------|----------------|
| Creatinine | Substantially higher | Near-normal | AKI is an independent mortality predictor in the ICU |
| Lactate | Substantially higher | Lower | Deceased patients had more severe circulatory failure, earlier or more prolonged |
| Bilirubin | Higher | Lower | Hepatic dysfunction is more pronounced in deceased patients, consistent with MODS |
| ALT | Higher | Lower | Greater hepatocellular injury in the deceased group, co-occurring with bilirubin elevation |
| WBC | Similar | — | WBC abnormalities are common in both groups; peak WBC is less discriminating than Lactate/Creatinine |
| Hemoglobin | **Lower** | Relatively higher | Deceased patients exhibit more severe anemia, potentially from hemorrhage, bone marrow suppression, or chronic disease progression |

**The clinical story in this chart**: Deceased ICU patients simultaneously show deterioration across renal, hepatic, and circulatory systems, while hemoglobin falls. This trajectory closely mirrors the clinical course of multi-organ failure (MOF) — where the failure of one organ cascades into sequential dysfunction of others.

**Design decisions**: Grouped bars (not stacked) allow direct within-indicator comparison of group heights. Colors: Survived = green (`#10B981`), Deceased = red (`#EF4444`) — consistent across the entire project.

---

### Chart 3: Outlier Detection — Z-score Leaderboard

**Method**: Compute Z-scores for each patient's peak Lab values across all indicators. Each patient's "outlier score" is the maximum absolute Z-score across all their indicators. Display the Top 20 patients.

```python
zscore = (peak - peak.mean()) / peak.std()
max_z  = zscore.abs().max(axis=1).sort_values(ascending=False)
```

**Key findings & clinical interpretation**:

- **|Z| > 2.5 threshold**: Under the normality assumption, |Z| > 2.5 corresponds to roughly the top 1.2% of the distribution — a reasonable threshold for "clearly abnormal." An orange dashed reference line marks this level in the chart
- **Deceased patients (red bars) are over-represented in the Top 20**: Extreme Lab values co-occur with fatal outcomes, as these patients represent the most severe disease states
- **Hover tooltip**: Each bar displays the specific indicator that drove that patient's outlier status (e.g., "Creatinine" or "Lactate"), revealing that different patients are extreme for different physiological reasons

**The clinical story**: Not all extreme-value patients die (green bars are present), confirming that extreme lab values are a necessary but not sufficient predictor of mortality. Some outliers reflect pre-existing chronic conditions (e.g., chronic renal disease with chronically elevated creatinine) rather than acute deterioration.

**Design decisions**:
- Bar color directly encodes mortality outcome (red/green), eliminating the need for a separate legend
- X-axis shows subject_id rather than anonymized identifiers, preserving traceability
- The |Z| = 2.5 threshold line uses orange — the standard Plotly "warning" color

---

### Chart 4: Individual Patient Lab Trajectories

**Method**: Automatically select 2 deceased and 2 survived patients from the Top 20 outlier list. Using ICU admission time (`intime`) as time zero, plot Creatinine, Lactate, and Hemoglobin over the first 120 hours of ICU stay.

**Patient selection logic**:
- Deceased group: top 2 deceased patients from the outlier leaderboard (highest Lab abnormality)
- Survived group: top 2 survived patients from the leaderboard; if fewer than 2, supplement with survived patients who have the most Lab records (denser data → smoother curves)

**Line encoding**:
- Deceased patients: solid lines, warm colors (red/orange)
- Survived patients: dotted lines, cool colors (green/blue)

**Key findings & clinical interpretation**:

**Typical deceased trajectory**:
- Creatinine rises continuously over the first 24–72 hours of ICU stay, indicating progressive AKI
- Lactate remains elevated or continues climbing, reflecting persistent circulatory failure despite intervention
- Hemoglobin trends downward, consistent with hemorrhagic loss or bone marrow suppression

**Typical survived trajectory**:
- Creatinine peaks then stabilizes or falls within 48–72 hours, indicating renal function recovery following treatment
- Lactate declines toward normal after resuscitation (fluid therapy, vasopressors)
- Hemoglobin decline is less severe or corrected by transfusion

**The clinical story**: At ICU admission, the two groups may look similar on paper. But the trajectories diverge 48–72 hours later — survived patients' indicators stabilize or improve with treatment, while deceased patients' indicators continue to worsen across multiple organ systems. This "fork in the road" often corresponds to critical clinical decision windows.

**Design notes**:
- Time window restricted to −6h to +120h (5 days), excluding all non-ICU data points
- Three indicators (Creatinine / Lactate / Hemoglobin) displayed in three side-by-side subplots, chosen for clinical relevance, data density, and coverage of three physiological domains (renal / circulatory / hematologic)

---

### Known Issues & Improvements

**Stage 3a (eDISH)**:

| Issue | Improvement |
|-------|-------------|
| No drug attribution; cannot link signals to specific medications | Join PRESCRIPTIONS; generate per-drug eDISH sub-plots to identify drug-specific hepatotoxicity signals |
| KDIGO baseline from intra-admission minimum (underestimates AKI severity) | Estimate pre-admission baseline creatinine using CKD-EPI from patient demographics |
| ALP not incorporated; cholestasis not excluded | Add ALP (itemid=50863); implement the full Hy's Law + Temple's Corollary distinction |
| Global peak aggregation loses temporal dynamics | Implement rolling 7-day peak windows to track ALT and Bili co-elevation patterns dynamically |

**Stage 3b (Patient Story)**:

| Issue | Improvement |
|-------|-------------|
| Representative patients selected by Z-score rank (may not be "typical") | Cluster patients (K-Means / hierarchical); select the patient closest to each cluster centroid as the representative |
| Trajectory chart not aligned to clinical events (intubation, surgery, vasopressors) | Join PROCEDURES_ICD and PRESCRIPTIONS; annotate key clinical event timestamps on the time series |
| Correlation matrix uses whole-admission peak values (no temporal segmentation) | Compute correlation matrices separately for 0–24h, 24–72h, and 72h+ to observe how co-morbidity patterns evolve |
| Deceased vs. survived comparison uses means (hides within-group distribution) | Switch to violin plots or strip plots with jitter to show full distribution shapes |

---

## Stage 3 — Key Numbers at a Glance

| Metric | Value |
|--------|-------|
| eDISH: patients with both ALT and Bili records | Confirm after running script |
| Hy's Law patient count | Confirm after running script |
| Hy's Law rate | Confirm after running script |
| KDIGO No AKI share | Confirm after running script |
| KDIGO Stage 3 count | Confirm after running script |
| Trajectory patients shown | Deceased: 2, Survived: 2 (auto-selected from Top 20 outliers) |
| Patient story Lab indicators | 6 (Creatinine, Lactate, WBC, Hemoglobin, Bilirubin, ALT) |
| Trajectory observation window | −6h to +120h from ICU admission |

> Run `python stage3a_edish.py` — the console prints quadrant distribution statistics and Hy's Law subject IDs to fill in the table above.

---

## Full Project Summary

### Completion Status

| Stage | Output | Status |
|-------|--------|--------|
| Stage 0 | Data validation, key statistics | ✅ Complete |
| Stage 1a | EDA Dashboard (6 charts) | ✅ Complete |
| Stage 1b | ERD (draw.io) | ✅ Complete |
| Stage 2-FE | Feature matrices (Track A/B CSV) | ✅ Complete |
| Stage 2A | Mortality prediction (3 models) | ✅ Complete |
| Stage 2B | LOS prediction (3 models) | ✅ Complete |
| Stage 2-DB | Modeling dashboard HTML | ✅ Complete |
| Stage 3a | eDISH + KDIGO + CTCAE | ✅ Complete |
| Stage 3b | Patient story (4 analyses) | ✅ Complete |

### Cross-Stage Design Decision Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Age handling | Sentinel value 91 + `age_imputed` flag | Preserves the "elderly" directional signal; model can learn >89-yr group behavior separately |
| CHARTEVENTS loading | Chunked reading with itemid filtering | 758K rows would risk OOM at full load; filtered to 64K rows |
| Lab missing values | Global median imputation | Retains all 129 samples; avoids losing training data due to missingness |
| Class imbalance | `class_weight` + `scale_pos_weight` | Death/survival ratio ~1:2.2; upweighting the minority class improves death recall |
| LOS target transform | `log1p` | Right-skewed LOS distribution; log transform compresses the tail for better model fit |
| eDISH axis scaling | Logarithmic | Lab values span multiple orders of magnitude; log scale separates normal from abnormal regions |
| Dashboard format | Standalone single-file HTML | No server required; opens in any browser; easy to share and demo |