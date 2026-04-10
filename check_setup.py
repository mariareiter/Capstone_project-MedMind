#!/usr/bin/env python3
"""
MedMind — pre-launch setup check

Run from the medmind_submission root to verify that all required files and
dependencies are in place.

Usage: python3 check_setup.py
"""
import os
import sys

GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN = "\033[96m"; BOLD = "\033[1m"; RESET = "\033[0m"
OK = f"{GREEN}✓{RESET}"; FAIL = f"{RED}✗{RESET}"; WARN = f"{YELLOW}!{RESET}"


def header(text):
    print(f"\n{BOLD}{CYAN}── {text} ──{RESET}")


def check_file(path, required=True):
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        print(f"  {OK} {path}  ({size_kb:.0f} KB)")
        return True
    marker = FAIL if required else WARN
    note = "(missing)" if required else "(optional, missing)"
    print(f"  {marker} {path}  {note}")
    return False


def check_import(module_name, display_name=None):
    name = display_name or module_name
    try:
        __import__(module_name)
        print(f"  {OK} {name}")
        return True
    except ImportError:
        print(f"  {FAIL} {name}  (install with: pip install {module_name})")
        return False


def main():
    print(f"\n{BOLD}MedMind — setup check{RESET}")
    print(f"Running from: {os.getcwd()}")
    all_ok = True

    # Python version
    header("Python version")
    py = sys.version_info
    if py >= (3, 10):
        print(f"  {OK} Python {py.major}.{py.minor}.{py.micro}")
    else:
        print(f"  {FAIL} Python {py.major}.{py.minor}.{py.micro}  (requires 3.10+)")
        all_ok = False

    # Core dependencies
    header("Python dependencies (required)")
    required_deps = [
        ("streamlit", "streamlit"), ("pandas", "pandas"), ("numpy", "numpy"),
        ("sklearn", "scikit-learn"), ("xgboost", "xgboost"),
        ("joblib", "joblib"), ("matplotlib", "matplotlib"),
    ]
    for mod, name in required_deps:
        if not check_import(mod, name):
            all_ok = False

    header("Python dependencies (optional)")
    optional_deps = [
        ("openai", "openai (MindGuide API mode)"),
        ("shap", "shap (notebook 03)"),
        ("seaborn", "seaborn (notebook 02)"),
        ("imblearn", "imbalanced-learn (notebook 03 SMOTE)"),
    ]
    for mod, name in optional_deps:
        try:
            __import__(mod)
            print(f"  {OK} {name}")
        except ImportError:
            print(f"  {WARN} {name}")

    # Repo structure
    header("Repository structure")
    for d in ["notebooks", "data", "models", "features", "results", "app", "docs"]:
        if os.path.isdir(d):
            print(f"  {OK} {d}/")
        else:
            print(f"  {FAIL} {d}/  (missing)")
            all_ok = False

    # Notebooks
    header("Notebooks")
    for nb in ["notebook_01_preprocessing.ipynb", "notebook_02_eda.ipynb", "notebook_03_modelling.ipynb"]:
        if not check_file(f"notebooks/{nb}"):
            all_ok = False

    # App
    header("App files")
    if not check_file("app/streamlit_app.py"):
        all_ok = False
    check_file("README.md", required=False)
    check_file(".streamlit/config.toml", required=False)

    # Dataset
    header("Dataset")
    if not check_file("data/dabe_clean_full.csv"):
        all_ok = False
        print(f"    {YELLOW}↳ Generate via notebooks/notebook_01_preprocessing.ipynb{RESET}")

    # Models
    header("Model files")
    outcomes = ["depression", "burnout", "anxiety", "suicidal_ideation"]
    variants = ["A", "B"]
    missing_mdl = False
    for o in outcomes:
        for v in variants:
            if not check_file(f"models/model_{o}_{v}.joblib"):
                all_ok = False
                missing_mdl = True
    if not check_file("models/thresholds.joblib"):
        all_ok = False
        missing_mdl = True
    if missing_mdl:
        print(f"    {YELLOW}↳ Generate via notebooks/notebook_03_modelling.ipynb{RESET}")

    # Feature lists
    header("Feature lists")
    missing_ft = False
    for o in outcomes:
        for v in variants:
            if not check_file(f"features/features_{o}_{v}.txt"):
                all_ok = False
                missing_ft = True
    if missing_ft:
        print(f"    {YELLOW}↳ Generate via notebooks/notebook_03_modelling.ipynb{RESET}")

    # Results CSVs (optional)
    header("Results CSVs (optional)")
    for f in ["tuning_results.csv", "full_vs_screening.csv",
              "bootstrap_confidence_intervals.csv", "threshold_analysis.csv",
              "smote_comparison.csv", "calibration_results.csv",
              "fairness_gender.csv", "thresholds.csv"]:
        check_file(f"results/{f}", required=False)

    # Environment
    header("Environment")
    if os.environ.get("OPENAI_API_KEY"):
        key = os.environ["OPENAI_API_KEY"]
        masked = key[:7] + "..." + key[-4:] if len(key) > 12 else "(set)"
        print(f"  {OK} OPENAI_API_KEY set  ({masked})")
    else:
        print(f"  {WARN} OPENAI_API_KEY not set  (MindGuide will use rule-based fallback)")

    if os.path.exists(".env"):
        print(f"  {OK} .env file present")
    else:
        print(f"  {WARN} .env file not present  (copy .env.example to .env and fill in key)")

    print()
    if all_ok:
        print(f"{BOLD}{GREEN}✓ All required components found.{RESET}")
        print(f"\nLaunch the app with:")
        print(f"  {CYAN}cd app && python3 -m streamlit run streamlit_app.py{RESET}\n")
        return 0

    print(f"{BOLD}{RED}✗ Some required components are missing.{RESET}")
    print(f"\nTo generate missing data/models/features, run the notebooks in order:")
    print(f"  1. notebooks/notebook_01_preprocessing.ipynb")
    print(f"  2. notebooks/notebook_02_eda.ipynb")
    print(f"  3. notebooks/notebook_03_modelling.ipynb\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
