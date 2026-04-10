# results/

This folder contains the CSV outputs from `notebooks/notebook_03_modelling.ipynb`. These files supply the performance tables referenced throughout the thesis Results chapter and the operating points used by the Streamlit application.

## Expected contents

**`tuning_results.csv`** — Bayesian hyperparameter search results for all four outcomes. Columns: `outcome`, `trial_id`, `learning_rate`, `max_depth`, `n_estimators`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `cv_auc_mean`, `cv_auc_std`. Used to generate Table 4.2 in the thesis.

**`full_vs_screening.csv`** — Head-to-head comparison of Model A (42 features) vs Model B (35 screening items) across all four outcomes. Columns: `outcome`, `variant`, `cv_auc_mean`, `cv_auc_std`, `test_auc`, `test_precision`, `test_recall`, `test_f1`, `delta_auc`. Used to generate Table 4.3 and the accompanying discussion of deployment realism.

**`bootstrap_confidence_intervals.csv`** — 95% bootstrap confidence intervals on all reported AUCs, computed on 1,000 resamples. Columns: `outcome`, `variant`, `auc_mean`, `auc_lower_ci`, `auc_upper_ci`. Used to generate the error bars in Figure 4.5.

**`threshold_analysis.csv`** — F1 score, precision, and recall across the full range of classification thresholds for each model. Columns: `outcome`, `variant`, `threshold`, `precision`, `recall`, `f1`, `specificity`. Used to select the operating points in `thresholds.csv` and to generate Figure 4.7.

**`smote_comparison.csv`** — Comparison of model performance with and without SMOTE oversampling on the minority class (particularly relevant for the rare suicidal ideation target). Columns: `outcome`, `variant`, `resampling`, `cv_auc_mean`, `cv_auc_std`, `minority_recall`. Used to justify the decision not to use SMOTE in the final models.

**`calibration_results.csv`** — Platt scaling calibration diagnostics: expected calibration error (ECE), maximum calibration error (MCE), and reliability curves for all eight models before and after calibration. Columns: `outcome`, `variant`, `stage` (pre/post calibration), `ece`, `mce`.

**`fairness_gender.csv`** — Exploratory fairness audit comparing model performance across gender subgroups. Columns: `outcome`, `variant`, `gender`, `n_samples`, `auc`, `precision`, `recall`, `demographic_parity_difference`, `equalised_odds_difference`. Used to support the discussion in Chapter 5.8.5.

**`thresholds.csv`** — The final F1-optimised operating points used by the Streamlit application, matching the content of `../models/thresholds.joblib`. Columns: `outcome`, `variant`, `threshold`, `f1_at_threshold`. This file is human-readable; the joblib version is what the app loads at runtime.

## How to generate

Run all cells of `notebooks/notebook_03_modelling.ipynb`. The notebook writes all eight CSV files to this folder.
