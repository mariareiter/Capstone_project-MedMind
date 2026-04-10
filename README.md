# MedMind

**From Risk to Action — An Explainable ML Screening Tool for Mental Health in Spanish Medical Students**

Author: **María Reiter Hernández**
IE University — Dual Degree in Business Administration + Data & Business Analytics (BDBA) and Computer Science + Artificial Intelligence (BCSAI)
Supervisor: Luis Angel Galindo
Capstone Project, 2026

---

## Overview

MedMind is an explainable machine learning screening system for depression, burnout, anxiety, and suicidal ideation in Spanish medical students. It is the first project to apply supervised ML to the DABE 2020 dataset (n = 5,216, across all 43 Spanish medical schools) and deploys the resulting models in a Streamlit application paired with an LLM-assisted conversational support agent (MindGuide).

The project delivers three things together. First, a reproducible analytical pipeline documented across three Jupyter notebooks that move from raw data to preprocessed dataset, from descriptive statistics to thesis figures, and from baseline models to tuned, explained, audited predictors. Second, eight trained XGBoost classifiers — two per outcome, covering a full 42-feature variant and a 35-item screening-only variant suitable for routine institutional deployment without clinical instruments. Third, a dual-audience web application that turns the models into a tool students and institutions can actually use: a 5-minute confidential screening for students with a unified results page and a supportive AI companion, and a population-level decision-support interface for wellbeing officers with SHAP rankings, costed intervention recommendations, and a red-flag triage dashboard.

The thesis argument is that mental health screening is only valuable if it translates into action. MedMind operationalises that claim in code.

## Model performance

Four binary classification outcomes, two feature-set variants per outcome, all 5-fold cross-validated on DABE 2020:

| Outcome            | Model A AUC (42 features) | Model B AUC (35 screening items) |
|:-------------------|:-------------------------:|:--------------------------------:|
| Depression         |           0.947           |              0.883               |
| Anxiety            |           0.881           |              0.822               |
| Suicidal ideation  |           0.878           |              0.814               |
| Burnout            |           0.765           |              0.731               |

Model A includes clinical instruments (STAI state anxiety, MBI emotional exhaustion) and is used for deep post-screening analysis. Model B uses only demographic, academic, social, life-event, substance-use, and perceived-difficulty items — no clinical questionnaires — and is the one that matters for realistic institutional deployment. Full tuning, calibration, threshold selection, bootstrap confidence intervals, SMOTE comparison, and fairness audit by gender are in `results/`.

---

## Repository structure

```
medmind_submission/
│
├── README.md                          ← This file
├── requirements.txt                   ← Python dependencies
├── LICENSE                            ← MIT + academic data notice
├── .gitignore                         ← Git exclusions
│
├── notebooks/
│   ├── notebook_01_preprocessing.ipynb   ← Data cleaning, feature engineering, targets
│   ├── notebook_02_eda.ipynb             ← Exploratory analysis and thesis figures
│   └── notebook_03_modelling.ipynb       ← Training, SHAP, fairness, final models
│
├── data/
│   └── dabe_clean_full.csv            ← Preprocessed dataset (output of notebook_01)
│
├── models/                            ← 8 trained model files + thresholds
│   ├── model_depression_A.joblib
│   ├── model_depression_B.joblib
│   ├── model_burnout_A.joblib
│   ├── model_burnout_B.joblib
│   ├── model_anxiety_A.joblib
│   ├── model_anxiety_B.joblib
│   ├── model_suicidal_ideation_A.joblib
│   ├── model_suicidal_ideation_B.joblib
│   └── thresholds.joblib              ← Operating points used by the app
│
├── features/                          ← 8 feature list files
│   ├── features_depression_A.txt
│   ├── features_depression_B.txt
│   ├── features_burnout_A.txt
│   ├── features_burnout_B.txt
│   ├── features_anxiety_A.txt
│   ├── features_anxiety_B.txt
│   ├── features_suicidal_ideation_A.txt
│   └── features_suicidal_ideation_B.txt
│
├── results/                           ← CSV outputs from notebook_03
│   ├── tuning_results.csv
│   ├── full_vs_screening.csv
│   ├── bootstrap_confidence_intervals.csv
│   ├── threshold_analysis.csv
│   ├── smote_comparison.csv
│   ├── calibration_results.csv
│   ├── fairness_gender.csv
│   └── thresholds.csv
│
├── app/
│   └── streamlit_app.py               ← The full Streamlit application
│
└── docs/
    ├── architecture.md                ← Technical overview of the app
    ├── getting-started.md             ← Step-by-step setup guide
    └── limitations.md                 ← Expanded limitations for thesis Chapter 5.8
```

---

## The notebooks

The three notebooks contain the full analytical pipeline behind the thesis. They are designed to run sequentially — each one takes the output of the previous as input — and together they reproduce every figure, table, and model referenced in the Results and Discussion chapters.

### `notebook_01_preprocessing.ipynb`

**Purpose.** Transform the raw DABE 2020 dataset into a clean, analysis-ready format with engineered features and binary classification targets.

**Inputs.** Raw DABE dataset (obtained separately from the authors of Capdevila-Gaudens et al. 2021; not redistributed in this repository).

**What it does.** Loads the raw survey responses, handles missing values, harmonises Likert scales across instruments, constructs composite variables (social support quality index, life events burden, academic dissatisfaction, effort-grade mismatch, number of perceived problems), recodes categorical variables for model consumption, and constructs the four binary classification targets using validated clinical cutoffs (Beck Depression Inventory ≥ 14 for depression, MBI exhaustion and cynicism subscale cutoffs for burnout, State-Trait Anxiety Inventory ≥ 40 for anxiety, any positive response on the suicidal ideation items). Applies inclusion criteria (complete responses, age range, consent to research use) and documents exclusions with counts at each stage.

**Outputs.** `data/dabe_clean_full.csv` — the preprocessed dataset used by notebooks 02 and 03 and by the Streamlit app's dashboard page.

**Key decisions documented in the notebook.** How to handle the subset of students who skipped the MBI but completed the BDI; the rationale for treating suicidal ideation as any positive response rather than attempting severity gradation; the decision to retain non-binary gender responses despite small cell counts (n = 18); the choice of BDI-II cutoff of 14 over the more conservative 20 to align with the thesis framing of preclinical elevated risk rather than clinical depression diagnosis.

### `notebook_02_eda.ipynb`

**Purpose.** Produce the descriptive findings and figures referenced in the Results chapter of the thesis.

**Inputs.** `dabe_clean_full.csv` (output of notebook 01).

**What it does.** Systematic exploratory analysis across demographics, target prevalence, score distributions, year-by-year progression, gender disaggregation, the social support dose-response relationship, life events, substance use, comorbidity patterns, and the preclinical-to-clinical transition. Each figure is annotated with the interpretive claim it supports in the thesis, and the numeric values behind each bar or line are printed alongside the plot so they can be copied directly into tables.

**Outputs.** 23 figures embedded in the notebook and referenced in Section 4.1 (Descriptive Findings) of the thesis. The notebook itself is the output — nothing is written to disk beyond the embedded plots.

**Key findings reproduced here.** The mid-degree vulnerability window for depression and anxiety (peak prevalence in Years 3-4, not Years 5-6 as expected). The monotonic worsening of burnout from Year 1 (22.6%) to Year 6 (44.8%) with a sharp 12-point jump at the preclinical-to-clinical transition. The dose-response relationship between social support and all four outcomes (students with 0 support persons show 64% depression prevalence; students with 5+ show 14%). The 5.5% of students scoring positive on all four outcomes simultaneously — a small but deeply vulnerable subgroup invisible to single-outcome screening programmes.

### `notebook_03_modelling.ipynb`

**Purpose.** Train, tune, evaluate, explain, and audit the eight classification models that power the Streamlit application.

**Inputs.** `dabe_clean_full.csv` (output of notebook 01).

**What it does.** This is the largest and most methodologically dense of the three notebooks. It proceeds in stages. First, baseline models across four algorithms (logistic regression, random forest, XGBoost, gradient boosting) to establish which family handles the data best. Second, Bayesian hyperparameter tuning of XGBoost with stratified 5-fold cross-validation, logged to `results/tuning_results.csv`. Third, the central methodological contribution: training two variants of each model in parallel — Model A with the full 42-feature set including clinical instruments, Model B with only 35 screening items that require no clinical expertise to collect — and comparing their performance head-to-head in `results/full_vs_screening.csv`. This comparison is what grounds the deployment realism of the whole thesis: can a 35-item questionnaire detect the same students that a full clinical battery detects? Fourth, bootstrap confidence intervals on 1,000 resamples for every reported AUC, written to `results/bootstrap_confidence_intervals.csv`. Fifth, a threshold analysis that selects the operating point for each model by optimising the F1 score on the validation fold, documented in `results/threshold_analysis.csv` and serialised as `models/thresholds.joblib` and `results/thresholds.csv` so the Streamlit app can use them directly. Sixth, a SMOTE comparison to test whether synthetic minority oversampling improves recall on the rare suicidal ideation target, results in `results/smote_comparison.csv`. Seventh, Platt scaling calibration to ensure predicted probabilities are interpretable, validated in `results/calibration_results.csv`. Eighth, SHAP analysis for every model, producing the feature importance rankings that feed directly into the Streamlit app's Explainable Risk Factors page. Ninth, a fairness audit by gender using demographic parity, equalised odds, and predictive parity metrics, written to `results/fairness_gender.csv` — the thesis explicitly discusses the limitations of this audit in Chapter 5.8.5.

**Outputs.** The eight trained models in `models/`, the eight feature lists in `features/`, the thresholds file, and the eight CSV files in `results/` that supply every performance table in the thesis. All SHAP plots and ROC curves are embedded in the notebook itself.

**Key methodological decisions documented in the notebook.** Why XGBoost over deep neural networks (small tabular data, explainability, deployment on modest hardware). Why SMOTE did not materially improve the suicidal ideation model despite class imbalance (the minority class patterns are too heterogeneous for synthetic augmentation). Why the threshold for suicidal ideation was deliberately set more conservatively than the F1-optimal point to minimise false negatives, at the cost of precision. Why the fairness audit is reported as exploratory rather than confirmatory (sample size limits statistical power on subgroup differences, and the outcome variables themselves may carry measurement bias across groups).

---

## The Streamlit application

The `app/streamlit_app.py` file (3,500+ lines) deploys the trained models as a dual-audience web application with 31 distinct pages organised into two role-specific hubs. The application is the applied contribution of the thesis and is what the defence demo will walk through.

**For students.** A confidential 5-minute screening predicts personal risk across all four outcomes. Results appear on a single unified page — risk cards for depression, burnout, and anxiety (suicidal ideation is never shown to students, a deliberate clinical safety decision documented below), SHAP-derived personal risk factors, a three-step action plan (this week, this month, ongoing), coping strategies automatically matched to the student's highest risk area, resource links to PAIME and the 024 national helpline, and a "Talk to MindGuide" button that opens a conversation with the supportive AI companion. Students can optionally sign in with a pseudonym — a username only they know, no email or real name — to save monthly tracker check-ins, sleep logs, letters to their future self, and MIR study plans across sessions. The student hub adapts to the user's screening results, reordering tools to highlight what's most relevant to their top risk, and includes an onboarding flow for first-time visitors and a nudge banner after 14 days of tracker inactivity.

**For institutions.** A complete decision-support interface for wellbeing officers, deans, and mental health programme administrators. Includes a population-level dashboard built on DABE 2020 (year-by-year prevalence, gender and phase breakdowns, risk distribution), a historical comparison of pre-pandemic DABE figures against post-COVID literature, a risk factor heatmap, a dedicated Explainable Risk Factors page that surfaces the top 5 SHAP contributors for each outcome with clinical interpretation and a cross-cutting factors panel, an intervention priority recommender that takes the institution's own data and produces a ranked, costed action plan with ROI estimates and SHAP-backed evidence, a cost-of-inaction calculator, a policy brief generator, a CSV upload tool for the institution's own screening data, a simulated reporting dashboard showing what a full semester of screening looks like in practice, a wellbeing officer triage dashboard with a red-flag workflow (cases move through new → contacted → in progress → resolved), an early warning system implementation guide, a programme evaluation framework, and a data collection protocol for contributing to external validation.

**MindGuide (the LLM companion).** MindGuide is a supportive AI chat interface accessible from the student hub and the results page. It operates in two modes depending on whether an OpenAI API key is available. With an API key, it uses GPT-4o-mini and receives a context-aware system prompt that includes the student's navigation history (which pages they have visited), so the conversation can reference what they have been exploring. Without an API key, it falls back to a rule-based response system that still handles the core supportive-reflection patterns — acknowledging distress, offering grounding techniques, suggesting concrete next steps, escalating to crisis resources when appropriate. No external network calls are made in fallback mode.

**Privacy architecture.** MedMind was built with a privacy-first design appropriate for mental health data. No accounts, no emails, no real names — students who want persistence pick a pseudonym. Screening responses and risk probabilities are held in Streamlit session state and cleared when the browser tab closes; they are never written to disk. Tracker entries, sleep logs, letters, and MIR plans are saved to `user_data/{pseudonym}.json` only when a student explicitly signs in, and that directory is gitignored. Institutional CSV uploads are processed entirely in memory and never touch disk. Suicidal ideation is never displayed to students: the SI model runs in the background, its output informs the MindGuide system prompt so the LLM can approach the conversation more carefully, and the soft crisis protocol surfaces the 024 national helpline and PAIME confidential physician support without ever displaying a number. This is an intentional clinical safety decision discussed in the thesis discussion chapter.

---

## Running the Streamlit application

### Requirements

- Python 3.10 or higher
- Approximately 500 MB of disk space for dependencies
- Optional: an OpenAI API key for the MindGuide conversational agent (falls back to rule-based responses if not provided)

### Installation

```bash
# 1. Clone the repository or unzip the submission
cd medmind_submission

# 2. Install dependencies
python3 -m pip install -r requirements.txt

# 3. (Optional) Set the OpenAI API key for MindGuide
export OPENAI_API_KEY="your-api-key-here"

# 4. Launch the application
cd app
python3 -m streamlit run streamlit_app.py
```

The application will open automatically in your default browser at `http://localhost:8501`.

**Note on file paths.** The Streamlit app automatically resolves the `models/`, `features/`, and `data/` folders as siblings of `app/` — no manual path configuration is needed. You can launch the app from either the project root or from inside the `app/` folder and it will find the required files in either case.

### Apple Silicon Macs

XGBoost requires the OpenMP runtime on Apple Silicon. If the depression model (or any other XGBoost model) fails to load with a libomp error, install it via Homebrew:

```bash
brew install libomp
```

---

## Running the notebooks

The three notebooks are designed to run sequentially in Google Colab, JupyterLab, or any notebook environment with Python 3.10+. To reproduce the full analysis from scratch:

1. Open `notebooks/notebook_01_preprocessing.ipynb` and run all cells. This takes the raw DABE dataset as input and produces `data/dabe_clean_full.csv`.
2. Open `notebooks/notebook_02_eda.ipynb` and run all cells. This produces the 23 exploratory figures referenced in the thesis Results chapter. Nothing is written to disk — the outputs are the figures embedded in the notebook itself.
3. Open `notebooks/notebook_03_modelling.ipynb` and run all cells. This produces the eight trained models (written to `models/`), the eight feature lists (written to `features/`), the thresholds file, and the eight CSV outputs in `results/`, plus all SHAP plots and performance figures embedded in the notebook.

**Important:** the notebooks expect the raw DABE 2020 dataset as input to notebook 01. This file is available from Capdevila-Gaudens et al. (2021) and is not redistributed in this repository. If you only need to inspect the analysis without re-running it, the notebooks include their executed cell outputs — you can read the full pipeline including all figures and tables without running anything.

---

## Privacy

The Streamlit application is designed to minimise data retention. Screening responses, risk predictions, and chat conversations with MindGuide are held in Streamlit session state and discarded when the browser tab is closed. Nothing is written to disk during a screening session.

The only exception is the optional pseudonym-based persistence feature: if a student chooses to sign in with a username, their tracker check-ins, sleep logs, letters, and MIR plans are saved to a local JSON file named after the pseudonym. This directory is gitignored and never transmitted anywhere. Students can use the application entirely anonymously without ever signing in.

Suicidal ideation risk scores are never displayed to students. The model runs in the background and informs crisis protocols, but no score is surfaced in the user interface.

---

## Citation

If you build on this work, please cite:

> Reiter Hernández, M. (2026). *MedMind: From Risk to Action — An Explainable ML Screening Tool for Mental Health in Spanish Medical Students*. Bachelor thesis, IE University, Madrid.

The underlying dataset is:

> Capdevila-Gaudens, P., García-Abajo, J. M., Flores-Funes, D., García-Barbero, M., & García-Estañ, J. (2021). Depression, anxiety, burnout and empathy among Spanish medical students. *PLoS ONE*, 16(12), e0260359. https://doi.org/10.1371/journal.pone.0260359

---

## License

Released under the MIT License — see `LICENSE` for details. The DABE 2020 dataset and any trained models derived from it are subject to the original dataset's academic use terms and should not be redistributed without permission from the original authors.

MedMind is not a medical device and is not intended to diagnose, treat, cure, or prevent any mental health condition. It is an educational and research tool developed as part of a Bachelor thesis at IE University. Users experiencing a mental health crisis should contact Spain's national suicide prevention helpline (024) or emergency services (112).

---

## Acknowledgements

Supervised by Luis Angel Galindo (IE University). Built on the DABE 2020 dataset collected by Capdevila-Gaudens and colleagues across all 43 Spanish medical schools. Crisis resources direct to Spain's national suicide prevention helpline (024) and the PAIME confidential physician support service operated by the Organización Médica Colegial de España.
