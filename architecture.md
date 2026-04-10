# MedMind — Architecture Overview

A quick reference for understanding how the application is structured. Useful for the thesis defence demo and for anyone reading the codebase for the first time.

## Technology stack

- **Frontend and server:** Streamlit (single-file Python app, ~3,500 lines)
- **Models:** XGBoost classifiers trained with scikit-learn pipelines, persisted as joblib files
- **Explainability:** SHAP values (precomputed from training, displayed as ranked features)
- **LLM companion:** OpenAI GPT-4o-mini via the official Python SDK, with rule-based fallback
- **Persistence:** Per-user JSON files under `user_data/{username}.json`, gitignored
- **Styling:** Custom CSS with Fraunces (serif, headlines) + Inter (sans, UI) from Google Fonts

## Page routing

All pages are rendered through a single `main()` function that dispatches on `st.session_state.page`. The routing table at the end of `streamlit_app.py` maps each page key to its rendering function. There are 31 pages in total.

```
gateway → student_hub → screening → results → agent (MindGuide)
                     ↘ coping, tracker, sleep_tracker, letter, stories,
                       exam_mode, benchmark, resources, mir_student

gateway → institution_hub → dashboard, historical, heatmap, shap_rankings,
                           playbook, calculator, policy_brief, earlywarning,
                           evaluation, csv_upload, intervention_recommender,
                           triage_dashboard, sim_dashboard, data_collection,
                           mir_institution, inst_benchmark
```

## Key architectural decisions

**Why a single file?** Streamlit re-runs the entire script on every interaction. Multi-file Streamlit apps require complex session state management to avoid re-importing and re-initialising on every run. A single file keeps the mental model simple and makes it easy to grep for anything.

**Why XGBoost over a deep neural network?** Three reasons: (1) the dataset is small (n=5,216) and tabular, which is XGBoost's ideal regime; (2) explainability is a core requirement, and SHAP on tree ensembles is both computationally cheap and provides theoretically sound contribution values; (3) clinical deployment requires models that can run on modest hardware without GPU infrastructure.

**Why two model variants (A and B)?** Model A uses the full 42-feature set including clinical instruments like STAI state anxiety and MBI emotional exhaustion. Model B uses only the 35 screening items that can be collected from a simple questionnaire, without requiring students to complete validated clinical instruments. Model A is used for deep analysis after a student has completed the full screening. Model B is the one that matters for routine institutional deployment — it answers the question "if we rolled this out to every student at semester start, what AUC could we realistically expect?"

**Why pseudonym-based profiles instead of accounts?** Mental health data is the most sensitive kind of personal data there is. Conventional account systems (email + password + real name) create a honeypot that is not justified by the user benefit. Pseudonyms deliver the same persistence benefit with none of the privacy cost — a student picks "bluefox42" as their username, their tracker history is saved to `user_data/bluefox42.json`, and nobody (including the app authors) can link that back to a real person.

**Why is suicidal ideation never shown to students?** Showing an SI risk score to a student in distress could be actively harmful. It could increase rumination, be misinterpreted as a prediction, or create a self-fulfilling framing. Instead, the SI model runs in the background, its output informs the MindGuide system prompt (so the LLM knows to approach the conversation more carefully), and the soft crisis protocol is triggered to surface the 024 helpline and PAIME support without ever displaying a number to the student. This is a conscious clinical safety decision, discussed at length in Chapter 5 of the thesis.

## Data flow

1. **Student arrives at gateway** → selects "I'm a student"
2. **Student hub** → if brand new, sees onboarding welcome card; otherwise sees the adaptive hub with their top risk area highlighted
3. **Optional: student signs in with pseudonym** → `render_profile_bar()` shows the sign-in field; on submit, `load_user_data(username)` hydrates `st.session_state` with any saved tracker, sleep, letter, MIR plan entries
4. **Student takes the 5-minute screening** → responses held in `st.session_state.screening_responses`
5. **Predictions generated** → `predict_all_outcomes()` loads all 8 models and computes probabilities; stores them in `st.session_state.screening_results`
6. **Results page** → unified display with risk cards, SHAP-derived factors, 3-step action plan, coping strategies matched to top risk, resource links, and a "Talk to MindGuide" button
7. **Student optionally talks to MindGuide** → `page_agent()` renders a chat interface; each message calls OpenAI (or falls back to rules) with a context-aware system prompt
8. **Student saves a tracker entry / letter / etc.** → if signed in, `save_user_data(username, key, value)` writes to `user_data/{username}.json`

## Security notes

- No outbound network calls except to OpenAI (optional, requires key) and Google Fonts (for the two fonts loaded by CSS `@import`)
- No telemetry, no analytics, no third-party scripts
- CSV upload tool processes files entirely in memory — uploaded data never touches disk
- Environment variables loaded via `python-dotenv` from `.env` file (gitignored)
- Streamlit's default `gatherUsageStats = true` is disabled in `.streamlit/config.toml`

## Known limitations

Documented in full in `docs/limitations.md` and in the thesis Chapter 5.8. Key points:

- **Circular validation.** The "screening-only" Model B is still predicting an outcome variable derived from the same screening questionnaire, which means the reported AUCs are inflated relative to what would be achieved predicting an independent clinical diagnosis.
- **No external validation.** All performance figures are cross-validated within DABE 2020. A different cohort — different era, different schools, different questionnaire — might show substantially lower AUCs.
- **SHAP is not causation.** The explainable risk factor rankings show what the model relies on, not what causes the outcome. Interventions targeting SHAP-identified factors are hypotheses to test, not guaranteed treatments.
- **No fairness audit.** Subgroup performance by gender, course year, and sexual orientation has been looked at but not formally audited with parity metrics. This is flagged as future work.
- **No deployment data.** Everything in the institutional side of the app — triage dashboard, intervention ROI estimates, cost calculator — is theoretical until a real institution pilots it. Implementation outcomes cannot be evaluated from a thesis prototype alone.

## File manifest

```
streamlit_app.py      ~3,500 lines — the whole application
README.md             project overview
requirements.txt      pinned Python dependencies
check_setup.py        pre-launch diagnostic
.gitignore            excludes user data, models, .env
.env.example          template for environment variables
.streamlit/config.toml Streamlit theme + server settings
docs/
  architecture.md     this file
  limitations.md      expanded limitations section for thesis Ch. 5.8
LICENSE               MIT + academic data notice
```
