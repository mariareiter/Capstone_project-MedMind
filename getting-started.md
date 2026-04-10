# Getting started

A quick-start guide for running MedMind for the first time. If you just need to get it working in 10 minutes, follow this.

## Step 1 — Prerequisites

You need:

- **Python 3.10 or later**. Check with `python3 --version`.
- **pip** (comes with Python).
- **Git** (if cloning from a repository).
- **Homebrew + libomp** (Apple Silicon Macs only, for XGBoost):
  ```
  brew install libomp
  ```

## Step 2 — Install dependencies

From the project root:

```
python3 -m pip install -r requirements.txt
```

This installs Streamlit, pandas, numpy, scikit-learn, XGBoost, joblib, matplotlib, openai, and python-dotenv.

## Step 3 — Obtain the data files

The repository does not include the DABE 2020 dataset or the trained models — they are subject to academic use terms. You need to place the following files in the project root:

**Dataset (2 files):**
- `dabe_clean_full.csv`
- `dabe_ml_features.csv`

**Models (8 files):**
- `model_depression_A.joblib`
- `model_depression_B.joblib`
- `model_burnout_A.joblib`
- `model_burnout_B.joblib`
- `model_anxiety_A.joblib`
- `model_anxiety_B.joblib`
- `model_si_A.joblib`
- `model_si_B.joblib`

**Feature lists (8 files):**
- `features_depression_A.joblib`
- `features_depression_B.joblib`
- `features_burnout_A.joblib`
- `features_burnout_B.joblib`
- `features_anxiety_A.joblib`
- `features_anxiety_B.joblib`
- `features_si_A.joblib`
- `features_si_B.joblib`

**Thresholds (1 file):**
- `thresholds.joblib`

**Logo (1 file, optional):**
- `medmind_logo.png` — if missing, the app falls back to an SVG placeholder

## Step 4 — (Optional) Configure MindGuide with an OpenAI key

MindGuide is the supportive AI companion accessible from the student hub. It works in two modes:

**With an API key** — uses GPT-4o-mini for context-aware supportive conversations.

**Without an API key** — falls back to a rule-based response system that still handles the core supportive-reflection patterns.

To enable the API mode, either export the key in your shell before launching:

```
export OPENAI_API_KEY=sk-your-key-here
```

Or create a `.env` file in the project root (copy from `.env.example`):

```
cp .env.example .env
# Then edit .env and fill in your key
```

## Step 5 — Run the setup check

Before launching, verify that everything is in place:

```
python3 check_setup.py
```

You should see a series of green check marks for Python version, dependencies, app files, model files, feature lists, thresholds, and dataset files. If any required component shows a red X, fix that issue before continuing. The OpenAI key and the logo file are optional — warnings for those are fine.

## Step 6 — Launch the app

```
python3 -m streamlit run streamlit_app.py
```

Streamlit will open the app in your default browser at `http://localhost:8501`. If it doesn't open automatically, copy the URL from the terminal output.

## What you should see

**First visit.** The MedMind landing page with a dark forest-to-navy header bar showing the MedMind logo, your name, and a "student" role badge. Below that, a cream-coloured hero section with the centred headline "Your mental health deserves better than a generic checklist" in serif type. Below the hero, two cards side by side: "I'm a student" with a sage-green icon on the left, "Institutional access" with a light-aqua icon on the right.

**Test the student flow.** Click "Enter as student" → you should see the student hub with an onboarding welcome card (first time only) → click "Take the 5-minute screening" → answer the questions → see your unified results page → click "Talk to MindGuide" to test the companion → return to the hub and explore the coping strategies, tracker, and peer stories.

**Test the institution flow.** Return to the gateway (click "MedMind" in the breadcrumb) → click "Enter as institution" → you should see the institution hub with four tabs: Data & analysis, Interventions & tools, Implementation, MIR support → explore the DABE dashboard, the Explainable risk factors page, the intervention recommender, and the wellbeing triage dashboard.

## Troubleshooting

**"XGBoost library not loaded" on Mac.** You need `libomp`:
```
brew install libomp
```

**"Module not found: streamlit" or similar.** Dependencies aren't installed:
```
python3 -m pip install -r requirements.txt
```

**"Dashboard data not found" when opening the institutional dashboard.** The DABE CSV is missing from the project root. Place `dabe_clean_full.csv` next to `streamlit_app.py` and refresh.

**Port 8501 already in use.** Something else is running on Streamlit's default port. Launch on a different port:
```
python3 -m streamlit run streamlit_app.py --server.port 8502
```

**Stale cached view after editing code.** Streamlit usually picks up file changes automatically, but if it doesn't, press `R` in the terminal running Streamlit to force a re-run, or restart the server.

**Pseudonym sign-in not persisting across restarts.** Check that the `user_data/` directory was created and contains a JSON file named after your username. If it's missing, check that the directory is writable.

**MindGuide gives generic rule-based responses instead of using the OpenAI API.** Either the `OPENAI_API_KEY` environment variable is not set, or the `openai` Python package is not installed. Run `python3 check_setup.py` to diagnose.
