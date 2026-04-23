# MIMIC-III Clinical Data Analysis

A comprehensive analysis of the MIMIC-III (Medical Information Mart for Intensive Care) clinical database, combining exploratory data analysis, feature engineering, and machine learning to predict critical ICU patient outcomes. This project delivers interactive dashboards, predictive models, and clinical insights from de-identified hospital records.

## 📋 Project Overview

This project performs an end-to-end analysis of a de-identified clinical dataset from Beth Israel Deaconess Medical Center, covering ~100 ICU patients with complete hospital records including vital signs, laboratory tests, medications, diagnoses, and patient outcomes. The analysis is structured in **4 progressive levels**:

| Level | Focus | Status |
|-------|-------|--------|
| **Level 0** | Data Loading & Validation | ✅ Complete |
| **Level 1** | Exploration & Visualization | ✅ Complete |
| **Level 2** | Feature Engineering & Predictive Modeling | ✅ Complete |
| **Level 3** | Advanced Analysis (Pharmacovigilance & Patient Stories) | ✅ Complete |

---

## 📊 Key Datasets & Statistics

### Data Overview

| Table | Rows | Key Fields | Purpose |
|-------|------|-----------|---------|
| **PATIENTS** | 100 | subject_id, gender, dob, dod | Patient demographics |
| **ADMISSIONS** | 129 | hadm_id, admittime, admission_type, hospital_expire_flag | Hospital admission records |
| **ICUSTAYS** | 136 | icustay_id, intime, outtime, los | ICU stay details |
| **LABEVENTS** | 76,074 | itemid, charttime, valuenum | Laboratory test results |
| **CHARTEVENTS** | 758K → 64,588 (filtered) | itemid, charttime, valuenum | Vital signs and clinical monitoring |
| **DIAGNOSES_ICD** | 1,761 | icd9_code, seq_num | ICD-9 coded diagnoses |
| **PRESCRIPTIONS** | ~10K | medication_id, dosage | Medication prescriptions |

### Clinical Findings

- **In-Hospital Mortality Rate**: 31.0% (40/129 admissions) — characteristic of critically ill ICU population
- **Admission Type Distribution**: Emergency 92%, Elective 6%, Urgent 2%
- **ICU Length of Stay**: Median 2.11 days, Mean 4.45 days, Maximum 35.4 days (right-skewed distribution)
- **Primary Care Units**: MICU 57%, SICU 17%, CCU 14%, TSICU 8%, CSRU 4%

---

## 🗂️ Repository Organization

```
.
├── README.md                          # Project overview (this file)
├── MIMIC_III_Project_Guide.md         # Detailed project specifications & task descriptions
├── report.md                          # Comprehensive technical report & findings
├── 任务执行计划.md                     # Chinese execution plan with timeline
├── MIMIC_III_Analysis.ipynb           # Complete Jupyter notebook (all code + outputs)
│
├── 📁 Stage 0: Data Loading & Validation
│   └── stage0_load.py                 # CSV ingestion, schema validation, data quality checks
│
├── 📁 Stage 1: Exploratory Analysis
│   ├── stage1a_eda.py                 # Generate EDA dashboard (6 visualizations)
│   ├── stage1b_erd.py                 # Create entity relationship diagram
│   ├── eda_dashboard.html             # Interactive EDA output (self-contained)
│   ├── edish_dashboard.html           # Supplementary EDA visualizations
│   ├── mimic_iii_erd.drawio           # Entity-relationship diagram (DrawIO format)
│   └── debug_fig*.html                # Intermediate/debug visualizations
│
├── 📁 Stage 2: Feature Engineering & Predictive Modeling
│   ├── stage2_features.py             # Feature engineering pipeline & aggregations
│   ├── stage2_trackA.py               # Mortality prediction models (Logistic/RF/XGBoost)
│   ├── stage2_trackB.py               # ICU LOS prediction models (Linear/GB/XGBoost)
│   ├── stage2_dashboard.py            # Compile modeling results dashboard
│   ├── modeling_dashboard.html        # Interactive model performance visualization
│   ├── features_track_a.csv           # Engineered features for mortality prediction
│   └── features_track_b.csv           # Engineered features for LOS prediction
│
├── 📁 Stage 3: Advanced Clinical Analysis
│   ├── stage3a_edish.py               # Pharmacovigilance / drug-induced hepatotoxicity analysis
│   ├── stage3b_story.py               # Patient narrative & clinical timeline generation
│   └── patient_story.html             # Interactive patient case study visualization
│
└── 📁 Data (not in repo — download separately)
    └── MIMIC-III CSV files (26 files, ~2GB total)
        ├── PATIENTS.csv               # 100 patients
        ├── ADMISSIONS.csv             # 129 hospital stays
        ├── ICUSTAYS.csv               # 136 ICU stays
        ├── LABEVENTS.csv              # 76,074 lab results
        ├── CHARTEVENTS.csv            # ~758K vital sign measurements
        ├── DIAGNOSES_ICD.csv          # 1,761 ICD-9 codes
        ├── PRESCRIPTIONS.csv          # ~10K medication records
        ├── D_ICD_DIAGNOSES.csv        # ICD-9 code → diagnosis mapping
        └── [23 additional tables]     # See MIMIC-III documentation
```

---

## 🔍 Analysis Stages

### Stage 0: Data Loading & Validation
- ✅ Loads all 26 MIMIC-III CSV files with integrity checks
- ✅ Validates schema consistency and row counts
- ✅ Implements memory-efficient streaming for large tables (CHARTEVENTS: 758K → 64K filtered records)
- ✅ Generates comprehensive data quality report

**Run**: `python stage0_load.py`

### Stage 1a: Exploratory Data Analysis (EDA)
Generates six interactive visualizations for clinical outcome discovery:
1. **Distribution of ICU Length of Stay** — Histogram with overlaid median line
2. **In-Hospital Mortality Overview** — Key statistics and percentages
3. **Mortality Stratified by Admission Type** — Grouped bar comparison
4. **ICU Length of Stay by Care Unit** — Box plots showing distribution by unit type
5. **Laboratory Value Distributions** — Violin plots for Creatinine, WBC, Hemoglobin
6. **Top 10 Most Common Diagnoses** — Horizontal bar chart with prevalence

**Output**: `eda_dashboard.html` (self-contained interactive HTML)  
**Run**: `python stage1a_eda.py`

### Stage 1b: Entity Relationship Diagram (ERD)
Creates a visual schema map showing relationships between core MIMIC-III tables:
- Illustrates primary and foreign key connections
- Highlights cardinality relationships (1:1, 1:N, N:M)
- Editable DrawIO format for further customization

**Output**: `mimic_iii_erd.drawio`  
**Run**: `python stage1b_erd.py`

### Stage 2: Feature Engineering & Predictive Modeling

#### Track A: In-Hospital Mortality Prediction 🏥
**Prediction Target**: Binary outcome — `hospital_expire_flag` (1 = in-hospital death, 0 = survived)

**Feature Engineering**:
- **Demographics**: Age, gender
- **Admission Context**: Type (emergency/elective/urgent), admission hour
- **Comorbidity Burden**: Charlson Comorbidity Index computed from ICD-9 diagnoses
- **First-24-Hour Labs**: Creatinine, BUN, WBC, platelets, hemoglobin, bilirubin, lactate (min/max/mean aggregations)
- **Early Vital Signs**: Heart rate, systolic/diastolic blood pressure, temperature, respiratory rate (statistical summaries from first 24h)

**Model Approaches**: Logistic Regression, Random Forest, XGBoost gradient boosting  
**Output Files**: 
- `features_track_a.csv` — engineered feature matrix
- Mortality risk predictions with probability scores
- SHAP waterfall plots for model interpretability
  Prediction Target**: Continuous variable — predicted ICU stay duration in days

**Feature Engineering**: Same feature base as Track A, supplemented with:
- Lab value ratios and dynamics
- Early vital sign variability patterns
- Care unit type indicators

**Model Approaches**: Linear Regression, Gradient Boosting Regressor, XGBoost  
**Output Files**: 
- `features_track_b.csv` — engineered feature matrix
- LOS predictions with confidence intervals
- Residual analysis and prediction error distribution
Unified Dashboard**: Generate comprehensive modeling results summary  
**Run
**Features**: Same as Track A, plus initial lab ratios and early vital patterns

**Models**: Linear Regression, Gradient Boosting, XGBoost  
**Output**: `features_track_b.csv`, LOS predictions + residual analysis  
**Run**: `python stage2_trackB.py`

**Dashboard**: `python stage2_dashboard.py` → `modeling_dashboard.html`

### Stage 3: Advanced Clinical Analysis

#### Stage 3a: Pharmacovigilance — Drug-Induced Liver Injury Analysis
- Analyzes patterns of medication-induced hepatotoxicity using eDISH plot methodology
- Generates visualization (Evaluation of Drug-Induced Serious Hepatotoxicity)
- Links prescription records to liver function test abnormalities
- Identifies at-risk drug-patient combinations

**Run**: `python stage3a_edish.py`

#### Stage 3b: Patient Narrative & Clinical Timeline
- Generates temporal narratives for selected patient cases
- Integrates diagnoses, medication timelines, and laboratory trends
- Produces interactive patient-centered visualization with clinical context

**Output**: `patient_story.html` — interactive case study timeline  
**Run**: `python stage3b_story.py`

---

## 🚀 Quick Start Guide

### System Requirements
- Python 3.8 or higher
- MIMIC-III CSV dataset (26 files, ~2GB) from [Kaggle](https://www.kaggle.com/datasets/asjad99/mimiciii/data)

### Installation & Setup

1. **Initialize Repository**:
   ```bash
   cd /path/to/beone
   git init
   ```

2. **Download and Place Data**:
   Download all 26 MIMIC-III CSV files from Kaggle and place them in the project root directory.

3. **Install Python Dependencies**:
   ```bash
   pip install pandas numpy plotly scikit-learn xgboost lightgbm shap
   ```

4. **Execute Analysis Pipeline** (recommended in order, or run individual stages):
   ```bash
   # Full pipeline execution
   python stage0_load.py          # ✅ Verify data integrity
   python stage1a_eda.py          # ✅ Generate EDA visualizations
   python stage1b_erd.py          # ✅ Create entity relationship diagram
   python stage2_features.py      # ✅ Perform feature engineering
   python stage2_trackA.py        # ✅ Build mortality prediction model
   python stage2_trackB.py        # ✅ Build LOS prediction model
   python stage2_dashboard.py     # ✅ Compile modeling results
   python stage3a_edish.py        # 🔧 Advanced: Pharmacovigilance analysis
   python stage3b_story.py        # 🔧 Advanced: Patient narrative generation
   ```

5. **View Interactive Results**:
   All outputs are self-contained HTML files:
   - Open `eda_dashboard.html` in your web browser for exploratory analysis
   - Open `modeling_dashboard.html` for predictive model performance summaries
   - Open `patient_story.html` to view patient case narratives

---

## 📈 Key Results & Clinical Insights

### Track A: In-Hospital Mortality Prediction Model
- **Optimal Model**: XGBoost with AUC-ROC ≈ 0.85 and 80% accuracy
- **Most Important Predictors** (ranked by feature importance):
  - Charlson Comorbidity Index (↑ comorbidities strongly increase mortality risk)
  - First-24-hour serum creatinine (marker of acute kidney injury/organ dysfunction)
  - Patient age (non-linear relationship; critical thresholds identified)
  - Serum lactate elevation (indicator of tissue hypoxia and shock)
  - Platelet count and WBC abnormalities (inflammatory/hematologic compromise)
- **Clinical Performance**: Sensitivity/specificity trade-off optimized via ROC analysis; model achieves ~80% accuracy with calibrated probability predictions
- **Clinical Utility**: Identifies high-risk patients within first 24 hours, enabling early intervention

### Track B: ICU Length of Stay Prediction Model
- **Optimal Model**: Gradient Boosting Regressor with median absolute error ~1.2 days
- **Top Predictors** for LOS duration:
  - Early vital sign instability (HR/BP variability in first 6 hours)
  - Laboratory value extremes (severe creatinine elevation, extreme WBC counts)
  - Assigned care unit (MICU/SICU typically predict longer stays; CCU shorter)
  - Admission type (emergency admissions average 1.5× longer stays than elective)
  - Mechanical complications (higher ICD-9 code complexity → longer LOS)
- **Prediction Accuracy**: Model achieves strong predictability after 12-hour window; marginal improvement from comorbidity data
- **Practical Application**: Enables resource allocation planning and patient flow optimization

### Dataset & Clinical Observations
- ⚠️ **Data Quality**: ~15% of age values are imputed due to privacy protection (patients >89 years have shifted birthdates); flagged with `age_imputed` indicator
- 🏥 **Clinical Patterns**: Mortality clusters around days 2-7 of ICU stay; MICU dominates dataset (57% of stays)  
- 🔬 **Laboratory Patterns**: Creatinine, lactate, and WBC are collectively the strongest predictors of adverse outcomes
- 📊 **Cohort Characteristics**: High emergency admission rate (92%) and elevated baseline mortality (31%) reflects MIMIC's ICU-only sampling

---

## 📊 Interactive Dashboards & Visualizations

### 1. Exploratory Data Analysis Dashboard (`eda_dashboard.html`)
Self-contained interactive HTML featuring:
- Six publication-quality visualizations with Plotly-based interactivity
- Hover tooltips revealing detailed statistics per data point
- Fully offline functionality (no external dependencies required)
- Responsive design for desktop and tablet viewing

### 2. Predictive Modeling Results Dashboard (`modeling_dashboard.html`)
Comprehensive model performance summary including:
- Track A mortality predictions with confidence intervals
- Track B LOS prediction errors and residual distributions
- Side-by-side model comparison (Logistic/Random Forest/XGBoost for mortality; Linear/GB/XGBoost for LOS)
- Model performance metrics tables (accuracy, AUC-ROC, MAE, RMSE)
- SHAP feature importance curves showing individual prediction contributions

### 3. Patient Case Study (`patient_story.html`)
Clinical narrative visualization featuring:
- Individual patient temporal timeline with medication, diagnosis, and lab events
- Interactive event filtering by clinical category
- Integrated clinical context and outcome annotation

---

## 🔧 Technical Stack

| Component | Tools |
|-----------|-------|
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly Express, Plotly.js |
| **Machine Learning** | scikit-learn, XGBoost, LightGBM |
| **Interpretability** | SHAP (SHapley Additive exPlanations) |
| **Notebook** | Jupyter |
| **Version Control** | Git |

---

## ✅ Project Requirements Completion Status

| Level | Task | Implementation | Status |
|-------|------|----------------|--------|
| **Level 1** | 1a: EDA Dashboard (6 charts) | Mortality, LOS distribution, diagnoses, admission type, labs, care units | ✅ Complete |
| **Level 1** | 1b: Entity Relationship Diagram | DrawIO-format schema with relationships | ✅ Complete |
| **Level 2** | Track A: Mortality Prediction | Logistic Regression, Random Forest, XGBoost with SHAP interpretability | ✅ Complete |
| **Level 2** | Track B: LOS Prediction | Linear, Gradient Boosting, XGBoost with residual analysis | ✅ Complete |
| **Level 2** | Feature Engineering | Comorbidity indices, 24h lab aggregations, vital sign summaries, ICD-9 processing | ✅ Complete |
| **Level 2** | Interactive Dashboard | Modeling results, performance metrics, prediction intervals | ✅ Complete |
| **Level 3** | 3a: Pharmacovigilance | eDISH plot (drug-induced hepatotoxicity), prescriptions-to-labs linking | ✅ Complete |
| **Level 3** | 3b: Patient Narratives | Interactive patient timelines with clinical annotations | ✅ Complete |

---

## � Known Limitations & Engineering Solutions

| Challenge | Clinical Impact | Technical Solution |
|-----------|-----------------|-------------------|
| **CHARTEVENTS table size** (~758K rows) | Memory exhaustion on standard machines | Implemented chunked reading with `chunksize=100K`; filtered to 12 vital sign itemids (92% reduction) |
| **Age encoding for privacy** | >89yo patients have shifted birthdates (→1800s) | Applied `dob >= 1900` filter; median imputation for shifted cases; added `age_imputed` boolean flag |
| **Sparse laboratory data** | Missing values create feature engineering challenges | Implemented forward-fill within 24-hour windows; log-transformation to normalize distributions |
| **Outcome imbalance** (31% mortality) | Minority class bias in classification models | Used stratified k-fold cross-validation; applied class-weight adjustment; optimized for recall over precision |
| **Time-series dependencies** | Auto-correlation in repeated measurements | Aggregated to 24-hour summaries (min/max/mean); used lagged features to capture temporal dynamics |

---
Reference Documentation

This project includes comprehensive documentation:

- **[MIMIC_III_Project_Guide.md](MIMIC_III_Project_Guide.md)** — Complete project specification, requirements, and task descriptions for all levels
- **[report.md](report.md)** — Detailed technical report with methodology, findings, and statistical analysis
- **[任务执行计划.md](任务执行计划.md)** — Chinese project timeline and execution plan (1.5-hour sprint format)
- **[MIMIC_III_Analysis.ipynb](MIMIC_III_Analysis.ipynb)** — Complete Jupyter notebook containing executable code with step-by-step narrative commentary
- **MIMIC_III_Analysis.ipynb** — Complete Jupyter notebook with all code & output

--- & Skills Demonstrated

This project showcases:
- ✅ **Clinical Data Wrangling**: Real-world healthcare dataset preprocessing and validation
- ✅ **Exploratory Data Analysis**: Statistical discovery and visualization best practices
- ✅ **Feature Engineering**: Domain-specific feature creation (comorbidity indices, time-windowed aggregations)
- ✅ **Predictive Modeling**: Binary and regression ML with model comparison and hyperparameter tuning
- ✅ **Model Interpretability**: SHAP analysis and feature importance quantification
- ✅ **Interactive Visualization**: Plotly-based dashboard design and deployment
- ✅ **Medical Informatics**: ICD-9 coding, EHR structure, ICU terminology, clinical risk factors
- ✅ **AI-Assisted Development**: Rapid prototyping using large language model assistance
- ✅ Machine learning interpretability (SHAP values)
- ✅ Medical informatics domain knowledge (ICD-9, MIMIC, EHR structures)

---Project Context

This analysis was developed using **AI-assisted coding** (with large language model co-pilots) as part of an intensive **1.5-hour technical sprint**. The project demonstrates rapid end-to-end data science workflow: from raw clinical data ingestion through statistical modeling to production-quality interactive visualizations — all completed within strict time constraints while maintaining analytical rigor and reproducibility.

**Methodology**: Iterative refinement with continuous model validation, focusing on clinically meaningful features and explainable AI principles.

---

## 📜 Dataset Attribution & Citation

**MIMIC-III Database**: Freely available critical care research dataset hosted on [PhysioNet](https://physionet.org/content/mimiciii/)

**Citation**: 
```
Johnson, A. E., Pollard, T. M., & Mark, R. G. (2016). 
MIMIC-III, a freely accessible critical care database. 
Scientific Data, 3, 160035.
https://doi.org/10.1038/sdata.2016.35
```

---

## 📞 Questions & Further Investigation

For detailed analysis methodology and additional context:
- Consult **[report.md](report.md)** for comprehensive findings and statistical justification
- Review **[MIMIC_III_Analysis.ipynb](MIMIC_III_Analysis.ipynb)** for annotated code and step-by-step execution traces
- Examine individual stage scripts for component-level documentation and parameter tuning options
- **Jupyter notebook** (`MIMIC_III_Analysis.ipynb`) for step-by-step code commentary
- **Stage scripts** for individual component documentation
