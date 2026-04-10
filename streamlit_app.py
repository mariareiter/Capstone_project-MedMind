"""
MedMind v2.1 — Mental Health Screening for Spanish Medical Students
Streamlit Application

Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import base64
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="MedMind — Mental Health Screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# LOGO
# ============================================================================

@st.cache_data
def load_logo_b64():
    # Try multiple locations: same dir as script, parent dir, and project root
    for p in ["./medmind_logo.png", "./logo.png", "../medmind_logo.png", "../assets/medmind_logo.png"]:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return base64.b64encode(f.read()).decode()
    return None

LOGO_B64 = load_logo_b64()

# ============================================================================
# CSS — all styles in one block
# ============================================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,400;0,500;0,600;1,400;1,500&family=Inter:wght@400;500;600;700&family=DM+Sans:wght@400;500;600&display=swap');

.stApp {
    font-family: 'DM Sans', sans-serif;
    background: #F5F0E4 !important;
    background-attachment: fixed !important;
}
[data-testid="stSidebar"] { display: none; }
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 0 !important; max-width: 1400px; }

/* ─── Top bar ─── */
.top-bar {
    background: linear-gradient(135deg, #1F3A2E 0%, #0F2B3D 50%, #0F2B3D 100%);
    padding: 24px 48px;
    display: flex; align-items: center; justify-content: space-between;
    margin: -1rem -5rem 0 -5rem;
    position: relative; z-index: 100;
    box-shadow: 0 4px 20px rgba(20,30,51,0.25);
}
.top-bar-left { display: flex; align-items: center; gap: 24px; }
.top-bar-logo { height: 88px; width: auto; border-radius: 12px; }
.top-bar h1 {
    font-family: 'Fraunces', serif;
    color: #fff; font-size: 38px; font-weight: 600; margin: 0;
    letter-spacing: -0.3px;
}
.top-bar .sub {
    font-family: 'Inter', sans-serif;
    color: rgba(255,255,255,0.45); font-size: 15px; margin-left: 16px; font-weight: 400;
}
.top-bar-right { display: flex; align-items: center; gap: 12px; }
.role-badge {
    font-size: 11px; font-weight: 500; letter-spacing: 0.6px;
    text-transform: uppercase;
    padding: 5px 16px; border-radius: 20px;
    background: rgba(255,255,255,0.08); color: rgba(255,255,255,0.65);
    border: 1px solid rgba(255,255,255,0.10);
}

/* ─── Breadcrumb ─── */
.breadcrumb {
    font-size: 12px; color: #8c96a4;
    padding: 14px 0 4px;
}
.breadcrumb a { color: #3d8baa; text-decoration: none; }
.breadcrumb a:hover { text-decoration: underline; }
.breadcrumb span.sep { margin: 0 6px; color: #c0c7d0; }

/* ─── Section header ─── */
.section-header {
    font-family: 'Fraunces', serif;
    font-size: 22px; font-weight: 500; color: #0F2B3D;
    margin: 32px 0 12px; padding-bottom: 10px;
    border-bottom: 1.5px solid #c8dbe8;
}

/* ─── Buttons ─── */
.stButton > button {
    background: #0F2B3D !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; padding: 14px 32px !important;
    font-weight: 500 !important; font-size: 16px !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.1px;
    box-shadow: 0 2px 8px rgba(27,42,74,0.12) !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #243759 !important;
    box-shadow: 0 4px 12px rgba(27,42,74,0.22) !important;
}

/* ─── Cards ─── */
.content-card {
    background: #fff; border: 1px solid #E5DCC5; border-radius: 14px;
    padding: 26px 28px; margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(27,42,74,0.04);
}
.content-card h4 {
    font-family: 'Fraunces', serif;
    font-size: 18px; font-weight: 500; color: #0F2B3D; margin-bottom: 10px;
}
.content-card p, .content-card li { font-size: 15px; color: #3a4a5e; line-height: 1.7; }

/* ─── Module cards ─── */
.mod-card {
    background: #fff; border: 1px solid #E5DCC5; border-radius: 10px;
    padding: 22px 20px; cursor: pointer;
    transition: all 0.15s; height: 100%;
    box-shadow: 0 1px 4px rgba(27,42,74,0.04);
}
.mod-card:hover {
    box-shadow: 0 6px 20px rgba(27,42,74,0.09);
    border-color: #c8d0dc;
}
.mod-card .mc-icon {
    width: 36px; height: 36px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 14px;
}
.mod-card h3 {
    font-family: 'Fraunces', serif;
    font-size: 15px; font-weight: 500; color: #0F2B3D; margin-bottom: 5px;
}
.mod-card p { font-size: 12px; color: #6b7a8d; line-height: 1.55; }

/* ─── Metric card ─── */
.metric-card {
    background: #fff; border: 1px solid #E5DCC5; border-radius: 10px;
    padding: 18px; text-align: center;
    box-shadow: 0 1px 4px rgba(27,42,74,0.04);
}
.metric-card .m-value { font-family: 'Fraunces', serif; font-size: 32px; font-weight: 600; color: #0F2B3D; }
.metric-card .m-label { font-size: 12px; color: #6b7a8d; text-transform: uppercase; letter-spacing: 0.4px; margin-top: 6px; }

/* ─── Risk cards ─── */
.risk-card {
    background: #fff; border-radius: 10px; padding: 22px 18px;
    text-align: center; border: 1px solid #E5DCC5;
    box-shadow: 0 1px 4px rgba(27,42,74,0.04);
}
.risk-card .label { font-size: 12px; color: #6b7a8d; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
.risk-card .value { font-family: 'Fraunces', serif; font-size: 36px; font-weight: 600; margin-bottom: 8px; }
.risk-card .level { font-size: 12px; font-weight: 500; padding: 5px 16px; border-radius: 12px; display: inline-block; }
.risk-high .value { color: #a32d2d; }
.risk-high .level { background: #fcebeb; color: #a32d2d; }
.risk-mod .value { color: #ba7517; }
.risk-mod .level { background: #faeeda; color: #854f0b; }
.risk-low .value { color: #0f6e56; }
.risk-low .level { background: #e1f5ee; color: #085041; }

/* ─── Factor bars ─── */
.factor-box {
    background: #fff; border: 1px solid #E5DCC5; border-radius: 10px;
    padding: 22px 24px; margin-top: 16px;
    box-shadow: 0 1px 4px rgba(27,42,74,0.04);
}
.factor-box h4 {
    font-family: 'Fraunces', serif;
    font-size: 16px; font-weight: 500; color: #0F2B3D; margin-bottom: 14px;
}

/* ─── AI box ─── */
.ai-box {
    background: #f2eef8; border: 1px solid #ddd5ea; border-radius: 10px;
    padding: 20px 22px; margin-top: 16px;
    display: flex; gap: 14px; align-items: flex-start;
}
.ai-icon {
    width: 34px; height: 34px; border-radius: 50%;
    background: #534ab7; color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 600; flex-shrink: 0;
}
.ai-text { font-size: 14px; color: #3a4a5e; line-height: 1.65; }
.ai-text strong { color: #0F2B3D; font-weight: 500; }

/* ─── Crisis box ─── */
.crisis-box {
    background: #f0e6f6; border: 1px solid #c9b3d9; border-radius: 10px;
    padding: 22px 24px; margin-top: 16px;
}
.crisis-box h4 {
    font-family: 'Fraunces', serif;
    font-size: 16px; color: #0F2B3D; font-weight: 500; margin-bottom: 8px;
}
.crisis-box p { font-size: 14px; color: #3a4a5e; line-height: 1.6; }
.crisis-box .hotline {
    display: inline-block; background: #0F2B3D; color: #fff;
    padding: 8px 20px; border-radius: 8px; font-size: 14px;
    font-weight: 500; margin-top: 10px; text-decoration: none;
}

/* ─── Story card ─── */
.story-card {
    background: #fff; border-left: 3px solid #3d8baa;
    border-radius: 0; padding: 20px 24px;
    margin-bottom: 12px; box-shadow: 0 1px 4px rgba(27,42,74,0.04);
}
.story-card .quote {
    font-family: 'Fraunces', serif;
    font-size: 16px; color: #0F2B3D; line-height: 1.75;
    font-style: italic; margin-bottom: 10px;
}
.story-card .attribution { font-size: 12px; color: #6b7a8d; font-style: normal; }

/* ─── Comparison table ─── */
.comp-table {
    width: 100%; border-collapse: separate; border-spacing: 0;
    font-size: 13px; border-radius: 8px; overflow: hidden; border: 1px solid #E5DCC5;
}
.comp-table th { background: #0F2B3D; color: #fff; padding: 11px 14px; text-align: left; font-weight: 500; font-size: 12px; }
.comp-table td { padding: 10px 14px; border-bottom: 1px solid #eef1f5; color: #3a4a5e; }
.comp-table tr:last-child td { border-bottom: none; }
.comp-table tr:nth-child(even) td { background: #f8f9fb; }
.trend-up { color: #a32d2d; font-weight: 600; }
.trend-same { color: #6b7a8d; }

/* ─── Coping header ─── */
.coping-header { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
.coping-header .badge {
    width: 36px; height: 36px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; background: #e4edf7;
}

/* ─── Footer ─── */
.app-footer {
    text-align: center; padding: 28px 0 12px; margin-top: 36px;
    border-top: 1px solid #E5DCC5;
}
.app-footer p { font-size: 11px; color: #a0a8b4; line-height: 1.6; }
.app-footer a { color: #3d8baa; text-decoration: none; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================================
# HELPERS
# ============================================================================

# Resolve sibling folders relative to this script's location, so the app works
# whether launched from the submission root or from the app/ subfolder itself.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR) if os.path.basename(_SCRIPT_DIR) == "app" else _SCRIPT_DIR
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
FEATURES_DIR = os.path.join(_PROJECT_ROOT, "features")
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

# Legacy fallback: if the sibling folders don't exist, assume flat layout.
if not os.path.isdir(MODELS_DIR):
    MODELS_DIR = _SCRIPT_DIR
if not os.path.isdir(FEATURES_DIR):
    FEATURES_DIR = _SCRIPT_DIR
if not os.path.isdir(DATA_DIR):
    DATA_DIR = _SCRIPT_DIR

@st.cache_resource
def load_models():
    models, features = {}, {}
    for target in ['depression', 'burnout', 'anxiety', 'suicidal_ideation']:
        for fs in ['A', 'B']:
            mp = os.path.join(MODELS_DIR, f"model_{target}_{fs}.joblib")
            fp = os.path.join(FEATURES_DIR, f"features_{target}_{fs}.txt")
            if os.path.exists(mp):
                try: models[f"{target}_{fs}"] = joblib.load(mp)
                except: pass
            if os.path.exists(fp):
                with open(fp) as f: features[f"{target}_{fs}"] = [l.strip() for l in f.readlines()]
    return models, features

@st.cache_data
def load_dashboard_data():
    p = os.path.join(DATA_DIR, "dabe_clean_full.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

def get_risk_level(prob):
    if prob >= 0.6: return "High", "risk-high"
    elif prob >= 0.3: return "Moderate", "risk-mod"
    return "Low", "risk-low"

def render_top_bar():
    if LOGO_B64:
        logo_html = f'<img src="data:image/png;base64,{LOGO_B64}" class="top-bar-logo" alt="MedMind">'
    else:
        logo_html = '<svg width="72" height="72" viewBox="0 0 48 48" fill="none" style="border-radius:14px;background:rgba(72,168,201,0.2);padding:10px;"><path d="M35.68 14.61a7.33 7.33 0 0 0-10.37 0L24 15.92l-1.31-1.31a7.33 7.33 0 0 0-10.37 10.37l1.31 1.31L24 36.66l10.37-10.37 1.31-1.31a7.33 7.33 0 0 0 0-10.37z" stroke="#48A8C9" stroke-width="1.8"/></svg>'
    role = st.session_state.get("role", "")
    role_label = "Student" if role == "student" else "Institution" if role == "institution" else ""
    role_badge = f'<span class="role-badge">{role_label}</span>' if role_label else ""
    st.markdown(f"""
    <div class="top-bar">
        <div class="top-bar-left">
            {logo_html}
            <h1>MedMind <span class="sub">Maria Reiter &middot; IE University</span></h1>
        </div>
        <div class="top-bar-right">{role_badge}</div>
    </div>""", unsafe_allow_html=True)
    # Profile bar appears on every student page
    render_profile_bar()

def render_breadcrumb(*crumbs):
    """Render a breadcrumb trail. Each crumb is (label, page_key) or just label for current."""
    parts = []
    for i, c in enumerate(crumbs):
        if isinstance(c, tuple):
            parts.append(f'<a href="javascript:void(0)">{c[0]}</a>')
        else:
            parts.append(f'<span style="color:#3a4a5e;font-weight:500;">{c}</span>')
    st.markdown(f'<div class="breadcrumb">{"<span class=sep>›</span>".join(parts)}</div>', unsafe_allow_html=True)

def render_footer():
    st.markdown("""
    <div class="app-footer">
        <p>MedMind — Capstone project by Maria Reiter Hernandez · IE University · 2026<br>
        Supervised by Luis Angel Galindo · Built on DABE 2020 data (n = 5,216)<br>
        Crisis support: <a href="tel:024">024</a> · <a href="https://www.icomem.es/seccion/SALUD-MENTAL-MEDICO" target="_blank">PAIME</a></p>
    </div>""", unsafe_allow_html=True)

def nav_back_to_hub():
    st.markdown("<br>", unsafe_allow_html=True)
    role = st.session_state.get("role", "student")
    label = "← Back to student hub" if role == "student" else "← Back to institution hub"
    target = "student_hub" if role == "student" else "institution_hub"
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button(label, key=f"back_{st.session_state.page}", use_container_width=True):
            st.session_state.page = target
            st.rerun()

def render_risk_card(label, prob):
    level, css_class = get_risk_level(prob)
    return f'<div class="risk-card {css_class}"><div class="label">{label}</div><div class="value">{prob*100:.0f}%</div><span class="level">{level}</span></div>'

# ── User persistence (JSON profiles) ──
USER_DATA_DIR = os.path.join(".", "user_data")

def save_user_data(username, data_key, data):
    """Save user data to a JSON file keyed by username."""
    os.makedirs(USER_DATA_DIR, exist_ok=True)
    safe_name = "".join(c for c in username.lower() if c.isalnum() or c == '_')
    path = os.path.join(USER_DATA_DIR, f"{safe_name}.json")
    try:
        existing = json.load(open(path)) if os.path.exists(path) else {}
    except: existing = {}
    existing[data_key] = data
    with open(path, 'w') as f: json.dump(existing, f, indent=2, default=str)

def load_user_data(username, data_key=None):
    """Load user data from JSON. Returns full dict or specific key."""
    safe_name = "".join(c for c in username.lower() if c.isalnum() or c == '_')
    path = os.path.join(USER_DATA_DIR, f"{safe_name}.json")
    try:
        data = json.load(open(path)) if os.path.exists(path) else {}
    except: data = {}
    return data.get(data_key) if data_key else data

def render_factor_bar(name, contribution, max_val=0.25):
    width = min(abs(contribution) / max_val * 100, 100)
    color = "#e24b4a" if contribution > 0.05 else "#ef9f27" if contribution > 0 else "#5dcaa5"
    sign = "+" if contribution > 0 else ""
    return f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;"><span style="font-size:12px;color:#3a4a5e;min-width:140px;">{name}</span><div style="flex:1;height:8px;background:#e8ecf1;border-radius:4px;overflow:hidden;"><div style="height:100%;width:{width:.0f}%;background:{color};border-radius:4px;"></div></div><span style="font-size:11px;color:#6b7a8d;min-width:40px;text-align:right;">{sign}{contribution*100:.0f}%</span></div>'

def generate_llm_response(risk_data):
    try:
        import anthropic
        client = anthropic.Anthropic()
        return client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=400,
            system="You are a compassionate mental health support assistant embedded in a university screening tool. Never diagnose. Focus on top 2-3 modifiable factors. Suggest concrete actions (PAIME at 024, counselling). Be warm and normalising. Under 100 words.",
            messages=[{"role": "user", "content": f"Student risk profile:\n{json.dumps(risk_data, indent=2)}\n\nGenerate personalised guidance."}]
        ).content[0].text
    except: return None

def get_fallback_guidance(risk_data):
    level = risk_data.get("highest_level", "moderate")
    g = f"Based on your responses, your profile shows {level} indicators that are common among medical students."
    top = risk_data.get("top_factors", [])
    if top:
        n = top[0].get("name", "")
        if "social_support" in n.lower(): g += " The strongest factor relates to social connections. Even small steps — reaching out, joining a study group — can help."
        elif "exhaustion" in n.lower(): g += " Exhaustion is playing a significant role. This should not be normalised as inevitable."
    if level == "high": g += " We recommend speaking with a professional. PAIME offers free support at 024."
    else: g += " Your university counselling service can provide additional support."
    return g


# ============================================================================
# SCREENING DEFINITIONS
# ============================================================================

SCREENING_SECTIONS = [
    {"title": "Demographics", "subtitle": "Basic information about you", "questions": [
        {"id": "Age", "text": "How old are you?", "type": "number", "min": 17, "max": 50, "default": 21},
        {"id": "gender_female", "text": "What is your gender?", "type": "select", "options": {"Male": 0, "Female": 1}},
        {"id": "course_year", "text": "What course year are you in?", "type": "select", "options": {"Year 1": 1, "Year 2": 2, "Year 3": 3, "Year 4": 4, "Year 5": 5, "Year 6": 6}},
        {"id": "orientation_non_hetero", "text": "Sexual orientation", "type": "select", "options": {"Heterosexual": 0, "Non-heterosexual": 1, "Prefer not to say": 0}},
        {"id": "works_any", "text": "Do you work alongside your studies?", "type": "select", "options": {"No": 0, "Yes": 1}},
        {"id": "has_fellowship", "text": "Do you have a fellowship or scholarship?", "type": "select", "options": {"No": 0, "Yes": 1}},
    ]},
    {"title": "Academic and social", "subtitle": "Your academic experience and social connections", "questions": [
        {"id": "attendance", "text": "How often do you attend non-compulsory classes?", "type": "slider", "min": 1, "max": 6, "default": 4, "labels": ["Never", "Always"]},
        {"id": "acad_effort_match", "text": "How well do your grades reflect your effort?", "type": "select", "options": {"Grades lower than effort": 1, "About right": 2, "Grades higher than effort": 3}},
        {"id": "acad_satisfaction_studies", "text": "How satisfied are you with your studies?", "type": "select", "options": {"Not at all": 1, "Somewhat": 2, "Quite": 3, "Very": 4}},
        {"id": "acad_satisfaction_choice", "text": "How satisfied are you with choosing medicine?", "type": "select", "options": {"Not at all": 1, "Somewhat": 2, "Quite": 3, "Very": 4}},
        {"id": "social_support_n", "text": "How many people can you count on for support?", "type": "select", "options": {"None": 1, "1-2 people": 2, "3-5 people": 3, "More than 5": 4}},
        {"id": "social_support_q2", "text": "Quality of emotional support from family/friends?", "type": "slider", "min": 1, "max": 5, "default": 3, "labels": ["None", "Excellent"]},
        {"id": "social_support_q3", "text": "Can you count on someone for practical help?", "type": "slider", "min": 1, "max": 5, "default": 3, "labels": ["Never", "Always"]},
        {"id": "social_support_q4", "text": "Do you have someone to talk to about worries?", "type": "slider", "min": 1, "max": 5, "default": 3, "labels": ["Never", "Always"]},
    ]},
    {"title": "Life events and wellbeing", "subtitle": "Recent experiences that may affect your wellbeing", "questions": [
        {"id": "eve_1", "text": "Serious illness or accident (yourself)?", "type": "select", "options": {"No": 0, "Yes": 1}},
        {"id": "eve_2", "text": "Serious illness in a family member?", "type": "select", "options": {"No": 0, "Yes": 1}},
        {"id": "eve_3", "text": "Serious financial difficulties?", "type": "select", "options": {"No": 0, "Yes": 1}},
        {"id": "eve_4", "text": "Relationship breakup?", "type": "select", "options": {"No": 0, "Yes": 1}},
        {"id": "eve_5", "text": "Death of someone close?", "type": "select", "options": {"No": 0, "Yes": 1}},
        {"id": "tobacco", "text": "Do you smoke?", "type": "select", "options": {"No": 0, "Occasionally": 1, "Regularly": 2}},
        {"id": "cannabis", "text": "Cannabis use?", "type": "select", "options": {"Never": 0, "Tried once": 1, "Occasionally": 2, "Regularly": 3}},
        {"id": "alcohol", "text": "How often do you drink alcohol?", "type": "select", "options": {"Never": 0, "Occasionally": 1, "1-2x/week": 2, "2-3x/week": 3, "3+/week": 4}},
        {"id": "takes_psychopharm", "text": "Currently taking psychiatric medication?", "type": "select", "options": {"No": 0, "Yes": 1}},
    ]},
    {"title": "Perceived difficulties", "subtitle": "Any areas where you're experiencing problems", "questions": [
        {"id": "prob_academic_performance", "text": "Academic performance", "type": "select", "options": {"No": 0, "Yes": 1}},
        {"id": "prob_family", "text": "Family issues", "type": "select", "options": {"No": 0, "Yes": 1}},
        {"id": "prob_daily_tasks", "text": "Managing daily tasks", "type": "select", "options": {"No": 0, "Yes": 1}},
        {"id": "prob_partner", "text": "Relationship problems", "type": "select", "options": {"No": 0, "Yes": 1}},
        {"id": "prob_time_management", "text": "Time management", "type": "select", "options": {"No": 0, "Yes": 1}},
        {"id": "prob_health", "text": "Health concerns", "type": "select", "options": {"No": 0, "Yes": 1}},
        {"id": "prob_money", "text": "Financial worries", "type": "select", "options": {"No": 0, "Yes": 1}},
        {"id": "prob_alcohol", "text": "Alcohol-related issues", "type": "select", "options": {"No": 0, "Yes": 1}},
        {"id": "prob_colleagues", "text": "Issues with colleagues", "type": "select", "options": {"No": 0, "Yes": 1}},
        {"id": "prob_tobacco", "text": "Tobacco-related issues", "type": "select", "options": {"No": 0, "Yes": 1}},
        {"id": "prob_teachers", "text": "Issues with teachers", "type": "select", "options": {"No": 0, "Yes": 1}},
        {"id": "prob_other_drugs", "text": "Other substance issues", "type": "select", "options": {"No": 0, "Yes": 1}},
    ]},
]


# ============================================================================
# UNIVERSITY RESOURCES DATABASE (expanded)
# ============================================================================

UNIVERSITY_RESOURCES = {
    # ─── Madrid ───
    "Universidad Complutense de Madrid (UCM)": {"c": "Clinica Universitaria de Psicologia — Campus Somosaguas. Free for UCM students.", "p": "91 394 31 11", "w": "https://www.ucm.es/clinica-universitaria-de-psicologia", "s": "AEMUCM runs peer support workshops.", "paime": "PAIME Madrid — ICOMEM. Tel: 91 538 51 00"},
    "Universidad Autonoma de Madrid (UAM)": {"c": "Oficina de Accion Solidaria — Psychological support, Campus Cantoblanco.", "p": "91 497 51 10", "w": "https://www.uam.es/uam/vida-uam/atencion-al-estudiante", "s": "CEEM delegations active in peer support.", "paime": "PAIME Madrid — ICOMEM. Tel: 91 538 51 00"},
    "Universidad Francisco de Vitoria (UFV)": {"c": "Servicio de Atencion Psicologica — Campus de Pozuelo. Free for UFV students, individual and group sessions available.", "p": "91 709 14 00", "w": "https://www.ufv.es/", "s": "Pastoral and student life services offer wellbeing programmes. Medical student mentorship active.", "paime": "PAIME Madrid — ICOMEM. Tel: 91 538 51 00"},
    "Universidad CEU San Pablo (Madrid)": {"c": "Gabinete de Orientacion Psicologica — Campus Monteprincipe.", "p": "91 456 63 00", "w": "https://www.uspceu.com/", "s": "CEU student associations run peer mentoring.", "paime": "PAIME Madrid — ICOMEM. Tel: 91 538 51 00"},
    "Universidad de Alcala (UAH)": {"c": "Unidad de Atencion Psicologica — free for UAH students.", "p": "91 885 50 00", "w": "https://www.uah.es/es/vivir-la-uah/servicios/", "s": "Medical student associations linked to CEEM.", "paime": "PAIME Madrid — ICOMEM. Tel: 91 538 51 00"},
    "Universidad Rey Juan Carlos (URJC)": {"c": "Clinica Universitaria — Campus de Alcorcon. Psychological support.", "p": "91 488 80 00", "w": "https://www.urjc.es/estudiar-en-la-urjc/vida-universitaria", "s": "Delegacion de Medicina runs study groups.", "paime": "PAIME Madrid — ICOMEM. Tel: 91 538 51 00"},
    "Universidad Europea de Madrid (UEM)": {"c": "Servicio de Bienestar Estudiantil — Campus Villaviciosa.", "p": "91 211 61 11", "w": "https://universidadeuropea.com/", "s": "Student life team organises wellbeing weeks.", "paime": "PAIME Madrid — ICOMEM. Tel: 91 538 51 00"},
    # ─── Catalonia ───
    "Universitat de Barcelona (UB)": {"c": "Servei d'Atencio a l'Estudiant (SAE) — Psychological support. Free.", "p": "93 402 16 47", "w": "https://www.ub.edu/web/ub/ca/estudians/sae/", "s": "AEMUB organises wellness events.", "paime": "PAIMM Catalunya — COMB. Tel: 93 567 88 88"},
    "Universitat Autonoma de Barcelona (UAB)": {"c": "Servei d'Assessorament i Formacio (SAF).", "p": "93 581 13 46", "w": "https://www.uab.cat/web/estudiar/", "s": "AEMB student group with wellness initiatives.", "paime": "PAIMM Catalunya — COMB. Tel: 93 567 88 88"},
    "Universitat Pompeu Fabra (UPF)": {"c": "Servei d'Atencio Psicologica — Campus Mar (shared with Hospital del Mar).", "p": "93 542 20 00", "w": "https://www.upf.edu/", "s": "UPF medical student association.", "paime": "PAIMM Catalunya — COMB. Tel: 93 567 88 88"},
    "Universitat de Lleida (UdL)": {"c": "Servei d'Atencio a l'Estudiant — Free psychological support.", "p": "973 70 20 00", "w": "https://www.udl.cat/", "s": "Student support through CEEM Lleida.", "paime": "PAIMM Catalunya — COMB. Tel: 93 567 88 88"},
    "Universitat de Girona (UdG)": {"c": "Servei d'Orientacio — Psychological counselling.", "p": "972 41 80 00", "w": "https://www.udg.edu/", "s": "Student associations offer peer support.", "paime": "PAIMM Catalunya — COMB. Tel: 93 567 88 88"},
    "Universitat Internacional de Catalunya (UIC)": {"c": "Student services — Psychological support programme.", "p": "93 254 18 00", "w": "https://www.uic.es/", "s": "Medical school wellbeing committee.", "paime": "PAIMM Catalunya — COMB. Tel: 93 567 88 88"},
    "Universitat Rovira i Virgili (URV)": {"c": "Servei d'Atencio a la Comunitat Universitaria.", "p": "977 55 80 00", "w": "https://www.urv.cat/", "s": "Reus campus medical student groups.", "paime": "PAIMM Catalunya — COMB. Tel: 93 567 88 88"},
    # ─── Valencia ───
    "Universidad de Valencia (UV)": {"c": "Unitat de Suport al Estudiantat — Servei de Psicologia. Free.", "p": "96 386 41 15", "w": "https://www.uv.es/", "s": "ADEITUV health committee.", "paime": "PAIME Valencia — COMV. Tel: 96 351 63 82"},
    "Universidad Miguel Hernandez (UMH)": {"c": "Servicio de Atencion Psicologica — Campus de Sant Joan d'Alacant.", "p": "96 591 90 00", "w": "https://www.umh.es/", "s": "Medical student association Sant Joan.", "paime": "PAIME Alicante — Colegio de Medicos de Alicante"},
    "Universidad Cardenal Herrera (CEU-UCH)": {"c": "Gabinete de Orientacion Psicologica — Campus de Moncada.", "p": "96 136 90 00", "w": "https://www.uchceu.es/", "s": "CEU peer tutoring programme.", "paime": "PAIME Valencia — COMV. Tel: 96 351 63 82"},
    "Universidad Jaume I (UJI)": {"c": "Servei d'Atencio a la Diversitat — Psychological support.", "p": "964 72 80 00", "w": "https://www.uji.es/", "s": "Student health committee.", "paime": "PAIME Castellon — Colegio de Medicos de Castellon"},
    # ─── Andalusia ───
    "Universidad de Granada (UGR)": {"c": "Gabinete Psicopedagogico — Vicerrectorado de Estudiantes. Free.", "p": "958 24 80 00", "w": "https://ve.ugr.es/", "s": "AEMGR runs exam-period support.", "paime": "PAIME Andalucia — Consejo Andaluz de Colegios de Medicos"},
    "Universidad de Sevilla (US)": {"c": "SACU — Servicio de Asistencia a la Comunidad Universitaria.", "p": "954 55 11 26", "w": "https://sacu.us.es/", "s": "AEMUS offers peer tutoring.", "paime": "PAIME Andalucia — Consejo Andaluz de Colegios de Medicos"},
    "Universidad de Cordoba (UCO)": {"c": "Unidad de Atencion Psicologica — free for UCO students.", "p": "957 21 80 00", "w": "https://www.uco.es/", "s": "Medical student association.", "paime": "PAIME Andalucia — Consejo Andaluz de Colegios de Medicos"},
    "Universidad de Malaga (UMA)": {"c": "Servicio de Atencion Psicologica — Campus Teatinos.", "p": "952 13 10 00", "w": "https://www.uma.es/", "s": "Student health and wellness committee.", "paime": "PAIME Andalucia — Consejo Andaluz de Colegios de Medicos"},
    "Universidad de Cadiz (UCA)": {"c": "Unidad de Atencion Psicologica y Pedagogica.", "p": "956 01 50 00", "w": "https://www.uca.es/", "s": "Medical school peer support.", "paime": "PAIME Andalucia — Consejo Andaluz de Colegios de Medicos"},
    "Universidad de Jaen (UJA)": {"c": "Gabinete de Psicologia — Free student service.", "p": "953 21 21 21", "w": "https://www.ujaen.es/", "s": "Student services wellbeing.", "paime": "PAIME Andalucia — Consejo Andaluz de Colegios de Medicos"},
    # ─── Basque Country / Navarra ───
    "Universidad del Pais Vasco (UPV/EHU)": {"c": "Servicio de Asesoramiento Educativo — Psychological support. Free.", "p": "94 601 20 00", "w": "https://www.ehu.eus/es/web/sae", "s": "Student council mental health committee.", "paime": "PAIME Pais Vasco — Colegio de Medicos de Bizkaia"},
    "Universidad de Navarra (UNAV)": {"c": "Servicio de Atencion Psicologica — Clinica Universidad de Navarra campus.", "p": "948 42 56 00", "w": "https://www.unav.edu/", "s": "AEUNAV medical student mentorship programme.", "paime": "PAIME Navarra — Colegio de Medicos de Navarra"},
    "Universidad de Deusto": {"c": "Servicio de Orientacion — Bilbao campus. Free for students.", "p": "94 413 90 00", "w": "https://www.deusto.es/", "s": "Student support network.", "paime": "PAIME Pais Vasco — Colegio de Medicos de Bizkaia"},
    # ─── Castilla y Leon ───
    "Universidad de Salamanca (USAL)": {"c": "Servicio de Asuntos Sociales — Psychological support unit. Free.", "p": "923 29 44 00", "w": "https://www.usal.es/", "s": "AEMUS-Salamanca runs peer support and exam workshops.", "paime": "PAIME Castilla y Leon — Colegio de Medicos de Salamanca"},
    "Universidad de Valladolid (UVA)": {"c": "Servicio de Atencion Psicologica — Campus Miguel Delibes.", "p": "983 42 30 00", "w": "https://www.uva.es/", "s": "Medical school student association.", "paime": "PAIME Castilla y Leon — Colegio de Medicos de Valladolid"},
    # ─── Galicia ───
    "Universidad de Santiago de Compostela (USC)": {"c": "Servizo de Participacion e Integracion Universitaria — Psychological support.", "p": "981 56 31 00", "w": "https://www.usc.gal/", "s": "AEMUSC peer support.", "paime": "PAIME Galicia — Colegio de Medicos de A Coruna"},
    # ─── Aragon ───
    "Universidad de Zaragoza (UNIZAR)": {"c": "Servicio de Orientacion y Empleo — Psychological counselling unit.", "p": "976 76 10 00", "w": "https://www.unizar.es/", "s": "Medical school student delegation.", "paime": "PAIME Aragon — Colegio de Medicos de Zaragoza"},
    # ─── Canarias / Murcia / Extremadura ───
    "Universidad de La Laguna (ULL)": {"c": "Servicio de Atencion Psicologica — Campus de Anchieta.", "p": "922 31 90 00", "w": "https://www.ull.es/", "s": "Student support in Tenerife.", "paime": "PAIME Canarias — Colegio de Medicos de Tenerife"},
    "Universidad de Las Palmas de Gran Canaria (ULPGC)": {"c": "Servicio de Orientacion — free psychological counselling.", "p": "928 45 10 00", "w": "https://www.ulpgc.es/", "s": "Medical student association.", "paime": "PAIME Canarias — Colegio de Medicos de Las Palmas"},
    "Universidad de Murcia (UM)": {"c": "Servicio de Atencion a la Diversidad y Voluntariado.", "p": "868 88 30 00", "w": "https://www.um.es/", "s": "AEMMU peer support.", "paime": "PAIME Murcia — Colegio de Medicos de Murcia"},
    "Universidad de Extremadura (UEX)": {"c": "Servicio de Orientacion — Campus de Badajoz. Free.", "p": "924 28 93 00", "w": "https://www.unex.es/", "s": "Student delegate network.", "paime": "PAIME Extremadura — Colegio de Medicos de Badajoz"},
    # ─── Cantabria / Asturias / La Rioja ───
    "Universidad de Cantabria (UC)": {"c": "Servicio de Atencion Psicologica — Campus de Santander.", "p": "942 20 10 00", "w": "https://web.unican.es/", "s": "AEMUC peer support programme.", "paime": "PAIME Cantabria — Colegio de Medicos de Cantabria"},
    "Universidad de Oviedo (UNIOVI)": {"c": "Centro de Orientacion — Psychological counselling. Free.", "p": "985 10 40 00", "w": "https://www.uniovi.es/", "s": "Student association wellness events.", "paime": "PAIME Asturias — Colegio de Medicos de Asturias"},
    # ─── Fallback ───
    "Other / My university is not listed": {"c": "Contact your university's student services — most Spanish universities offer free psychological support.", "p": "—", "w": "—", "s": "Check your faculty's CEEM-affiliated student association.", "paime": "Find your regional PAIME through your provincial Colegio de Medicos."},
}


# ============================================================================
# PAGE: GATEWAY (role selection)
# ============================================================================

def page_gateway():
    render_top_bar()

    hero_html = (
        '<div style="position:relative;overflow:hidden;padding:44px 0 36px;background:#F5F0E4;margin:0 -5rem 0 -5rem;padding-left:5rem;padding-right:5rem;">'
        '<div style="position:absolute;top:-120px;left:50%;transform:translateX(-50%);width:720px;height:380px;border-radius:50%;background:linear-gradient(180deg,#D6EDF2,transparent);opacity:0.55;pointer-events:none;"></div>'
        '<div style="position:absolute;bottom:-100px;left:-50px;width:280px;height:280px;border-radius:50%;background:#DCE9D7;opacity:0.35;pointer-events:none;"></div>'
        '<div style="position:absolute;top:20%;right:10%;width:140px;height:140px;border-radius:50%;border:1.5px solid rgba(31,58,46,0.08);pointer-events:none;"></div>'
        '<div style="position:relative;z-index:2;text-align:center;max-width:780px;margin:0 auto;">'
        '<div style="display:inline-flex;align-items:center;gap:12px;font-family:\'Inter\',sans-serif;font-size:12px;font-weight:600;color:#48A8C9;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:18px;">'
        '<span style="width:32px;height:1.5px;background:#48A8C9;display:inline-block;"></span>'
        'Spanish medical students · 5,216 surveyed'
        '<span style="width:32px;height:1.5px;background:#48A8C9;display:inline-block;"></span>'
        '</div>'
        '<p style="font-family:\'Fraunces\',serif;font-size:52px;font-weight:500;color:#0F2B3D;line-height:1.06;letter-spacing:-1.6px;margin-bottom:20px;">'
        'Your mental health deserves better than <span style="font-style:italic;font-weight:400;background:linear-gradient(135deg,#2E5244,#48A8C9);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">a generic checklist.</span>'
        '</p>'
        '<p style="font-family:\'Inter\',sans-serif;font-size:16px;color:#4A5560;line-height:1.6;max-width:620px;margin:0 auto 22px;">'
        'Screening built on 5,216 medical students across Spain. Confidential, explained, and designed for the realities of your training.'
        '</p>'
        '<div style="display:flex;gap:28px;flex-wrap:wrap;justify-content:center;margin-bottom:8px;">'
        '<div style="display:flex;align-items:center;gap:10px;">'
        '<div style="width:38px;height:38px;border-radius:11px;background:#fff;border:1px solid #E5DCC5;display:flex;align-items:center;justify-content:center;">'
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#48A8C9" stroke-width="1.8"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>'
        '</div>'
        '<div style="font-family:\'Inter\',sans-serif;font-size:13px;font-weight:500;color:#4A5560;line-height:1.3;text-align:left;">'
        '<span style="display:block;font-size:11px;color:#48A8C9;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">Privacy</span>by design'
        '</div>'
        '</div>'
        '<div style="display:flex;align-items:center;gap:10px;">'
        '<div style="width:38px;height:38px;border-radius:11px;background:#fff;border:1px solid #E5DCC5;display:flex;align-items:center;justify-content:center;">'
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#48A8C9" stroke-width="1.8"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>'
        '</div>'
        '<div style="font-family:\'Inter\',sans-serif;font-size:13px;font-weight:500;color:#4A5560;line-height:1.3;text-align:left;">'
        '<span style="display:block;font-size:11px;color:#48A8C9;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">Validated</span>instruments'
        '</div>'
        '</div>'
        '<div style="display:flex;align-items:center;gap:10px;">'
        '<div style="width:38px;height:38px;border-radius:11px;background:#fff;border:1px solid #E5DCC5;display:flex;align-items:center;justify-content:center;">'
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#48A8C9" stroke-width="1.8"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>'
        '</div>'
        '<div style="font-family:\'Inter\',sans-serif;font-size:13px;font-weight:500;color:#4A5560;line-height:1.3;text-align:left;">'
        '<span style="display:block;font-size:11px;color:#48A8C9;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">PAIME</span>'
        '<a href="https://www.icomem.es/seccion/SALUD-MENTAL-MEDICO" target="_blank" style="color:#4A5560;text-decoration:none;">affiliated</a>'
        '</div>'
        '</div>'
        '</div>'
        '</div>'
        '</div>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)

    # Cards row below hero
    col_s, col_i = st.columns(2, gap="medium")

    with col_s:
        student_card = (
            '<div style="background:#fff;border-radius:20px;padding:26px 28px 20px;border:1px solid #E5DCC5;position:relative;overflow:hidden;margin-top:12px;">'
            '<div style="position:absolute;top:0;right:0;width:140px;height:140px;border-radius:50%;background:#DCE9D7;opacity:0.35;transform:translate(35%,-35%);pointer-events:none;"></div>'
            '<div style="display:flex;align-items:center;gap:16px;margin-bottom:12px;position:relative;z-index:2;">'
            '<div style="width:56px;height:56px;border-radius:15px;background:#DCE9D7;display:flex;align-items:center;justify-content:center;flex-shrink:0;">'
            '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#2E5244" stroke-width="1.8"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>'
            '</div>'
            '<h3 style="font-family:\'Fraunces\',serif;font-size:24px;font-weight:500;color:#0F2B3D;letter-spacing:-0.4px;margin:0;line-height:1.1;">I\'m a student</h3>'
            '</div>'
            '<p style="font-family:\'Inter\',sans-serif;font-size:14px;color:#6C766F;line-height:1.55;margin-bottom:14px;position:relative;z-index:2;">Screening, strategies, and support that understand what you\'re going through.</p>'
            '</div>'
        )
        st.markdown(student_card, unsafe_allow_html=True)
        if st.button("Enter as student", key="gw_student", use_container_width=True):
            st.session_state.role = "student"
            st.session_state.page = "student_hub"
            st.rerun()

    with col_i:
        inst_card = (
            '<div style="background:#fff;border-radius:20px;padding:26px 28px 20px;border:1px solid #E5DCC5;position:relative;overflow:hidden;margin-top:12px;">'
            '<div style="position:absolute;top:0;right:0;width:140px;height:140px;border-radius:50%;background:#D6EDF2;opacity:0.35;transform:translate(35%,-35%);pointer-events:none;"></div>'
            '<div style="display:flex;align-items:center;gap:16px;margin-bottom:12px;position:relative;z-index:2;">'
            '<div style="width:56px;height:56px;border-radius:15px;background:#D6EDF2;display:flex;align-items:center;justify-content:center;flex-shrink:0;">'
            '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#0F2B3D" stroke-width="1.8"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>'
            '</div>'
            '<h3 style="font-family:\'Fraunces\',serif;font-size:24px;font-weight:500;color:#0F2B3D;letter-spacing:-0.4px;margin:0;line-height:1.1;">Institutional access</h3>'
            '</div>'
            '<p style="font-family:\'Inter\',sans-serif;font-size:14px;color:#6C766F;line-height:1.55;margin-bottom:14px;position:relative;z-index:2;">Data, interventions, and the tools to make the case for action.</p>'
            '</div>'
        )
        st.markdown(inst_card, unsafe_allow_html=True)
        if st.button("Enter as institution", key="gw_inst", use_container_width=True):
            st.session_state.role = "institution"
            st.session_state.page = "institution_hub"
            st.rerun()

    render_footer()


# ============================================================================
# PAGE: STUDENT HUB
# ============================================================================

def get_personalization_context():
    """Returns a dict describing the student's current state for adaptive content."""
    ctx = {
        "has_screening": False,
        "top_risk": None,
        "top_risk_level": None,
        "is_first_time": True,
        "days_since_last_tracker": None,
        "signed_in": bool(st.session_state.get("user_profile", "")),
        "username": st.session_state.get("user_profile", ""),
    }
    results = st.session_state.get("screening_results", {})
    if results:
        ctx["has_screening"] = True
        ctx["is_first_time"] = False
        vis = {k: v for k, v in results.items() if k != 'suicidal_ideation'}
        if vis:
            ctx["top_risk"] = max(vis, key=vis.get)
            p = vis[ctx["top_risk"]]
            ctx["top_risk_level"] = get_risk_level(p)[0].lower()
    if st.session_state.get("tracker_history"):
        ctx["is_first_time"] = False
        try:
            from datetime import datetime
            last = st.session_state.tracker_history[-1]["date"]
            last_dt = datetime.strptime(last, "%Y-%m-%d")
            ctx["days_since_last_tracker"] = (datetime.now() - last_dt).days
        except Exception: pass
    return ctx

def render_onboarding_or_nudge(ctx):
    """Renders the right welcome/nudge at the top of the student hub depending on state."""
    if ctx["is_first_time"] and not ctx["has_screening"] and not st.session_state.get("onboarding_skipped", False):
        # Brand new user — onboarding
        st.markdown("""<div style="margin-top:16px;padding:28px 28px;background:linear-gradient(135deg,#fff,#F5F9FE);border:1px solid #48A8C9;border-radius:16px;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
                <div style="width:32px;height:32px;border-radius:50%;background:#D6EDF2;display:flex;align-items:center;justify-content:center;">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#48A8C9" stroke-width="2"><path d="M12 2v20M2 12h20"/></svg>
                </div>
                <span style="font-size:11px;font-weight:600;color:#48A8C9;text-transform:uppercase;letter-spacing:0.5px;">Start here</span>
            </div>
            <h3 style="font-family:'Fraunces',serif;font-size:22px;font-weight:500;color:#0F2B3D;margin-bottom:8px;">Welcome to MedMind</h3>
            <p style="font-size:15px;color:#3a4a5e;line-height:1.7;margin-bottom:16px;max-width:640px;">
                MedMind is built for the realities of medical training. The fastest way to get value is to take the 5-minute screening — it unlocks personalised results, a targeted action plan, coping strategies matched to your profile, and contextual guidance throughout the app. Or, if you'd rather explore first, the tools below are all available without signing up.</p>
        </div>""", unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1])
        with c1:
            if st.button("Take the 5-minute screening", key="onb_screen", use_container_width=True, type="primary"):
                st.session_state.page = "screening"; st.rerun()
        with c2:
            if st.button("Skip — let me explore", key="onb_skip", use_container_width=True):
                st.session_state.onboarding_skipped = True; st.rerun()
        return True
    if ctx["has_screening"]:
        # Student has taken screening — show adaptive summary
        risk_label = {"depression": "depression", "burnout": "burnout", "anxiety": "anxiety"}.get(ctx["top_risk"], ctx["top_risk"] or "")
        level = ctx["top_risk_level"] or "low"
        level_color = {"high": "#c92f2f", "moderate": "#c78017", "low": "#0F6E56"}.get(level, "#0F6E56")
        greeting = f"Welcome back, {ctx['username']}" if ctx["signed_in"] else "Based on your screening"
        st.markdown(f"""<div style="margin-top:16px;padding:22px 24px;background:#fff;border:1px solid #E5DCC5;border-radius:14px;border-left:4px solid {level_color};">
            <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
                <div>
                    <span style="font-size:11px;font-weight:600;color:#888780;text-transform:uppercase;letter-spacing:0.5px;">{greeting}</span>
                    <h4 style="font-family:'Fraunces',serif;font-size:18px;font-weight:500;color:#0F2B3D;margin:4px 0 0;">Your highest-risk area is <span style="color:{level_color};">{risk_label}</span> ({level})</h4>
                    <p style="font-size:13px;color:#5a6a7e;margin:6px 0 0;">We've reordered your tools to put what's most relevant to you first.</p>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
    # Tracker nudge
    if ctx["days_since_last_tracker"] is not None and ctx["days_since_last_tracker"] >= 14:
        st.markdown(f"""<div style="margin-top:10px;padding:14px 20px;background:#FAEEDA;border:1px solid #FAC775;border-radius:12px;display:flex;align-items:center;gap:10px;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#854F0B" stroke-width="1.8" style="flex-shrink:0;"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
            <span style="font-size:13px;color:#854F0B;">It's been <strong>{ctx["days_since_last_tracker"]} days</strong> since your last tracker check-in. A 2-minute entry now helps you spot patterns.</span>
        </div>""", unsafe_allow_html=True)
    return False

def generate_tracker_ics():
    """Returns bytes of a .ics calendar file with a monthly recurring tracker reminder."""
    from datetime import datetime, timedelta
    start = (datetime.now() + timedelta(days=30)).strftime("%Y%m%dT090000")
    end = (datetime.now() + timedelta(days=30, hours=0, minutes=10)).strftime("%Y%m%dT091000")
    uid = f"medmind-tracker-{datetime.now().strftime('%Y%m%d%H%M%S')}@medmind.local"
    ics = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//MedMind//Monthly Tracker Reminder//EN
CALSCALE:GREGORIAN
METHOD:PUBLISH
BEGIN:VEVENT
UID:{uid}
DTSTAMP:{datetime.now().strftime("%Y%m%dT%H%M%S")}
DTSTART:{start}
DTEND:{end}
SUMMARY:MedMind monthly check-in
DESCRIPTION:Time for your 2-minute wellbeing check-in on MedMind. Log in at your usual username to save your entry to your profile.
RRULE:FREQ=MONTHLY;INTERVAL=1
BEGIN:VALARM
TRIGGER:-PT30M
ACTION:DISPLAY
DESCRIPTION:MedMind check-in in 30 minutes
END:VALARM
END:VEVENT
END:VCALENDAR
"""
    return ics.encode("utf-8")

def render_profile_bar():
    """Profile bar — visible on every student page. Only renders for student role."""
    if st.session_state.get("role", "") != "student":
        return
    current = st.session_state.get("user_profile", "")
    if current:
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(f"""<div style="background:#D6EDF2;border:1px solid #48A8C9;border-radius:12px;padding:12px 18px;display:flex;align-items:center;gap:10px;margin-top:12px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0F2B3D" stroke-width="1.8"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
                <span style="font-size:13px;color:#0F2B3D;">Signed in as <strong>{current}</strong> — your tracker, sleep log, letters, and MIR planner are saved to your profile.</span>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown('<div style="margin-top:12px;"></div>', unsafe_allow_html=True)
            if st.button("Switch profile", key="profile_switch", use_container_width=True):
                st.session_state.user_profile = ""
                for k in ["tracker_history", "sleep_log", "letter", "mir_plan"]:
                    if k in st.session_state: del st.session_state[k]
                st.rerun()
    else:
        st.markdown("""<div style="background:#FAEEDA;border:1px solid #FAC775;border-radius:12px;padding:14px 18px;margin-top:12px;">
            <p style="font-size:13px;color:#854F0B;margin:0 0 8px;line-height:1.5;">
                <strong>Save your progress.</strong> Pick a username to keep your tracker, sleep log, letters, and MIR planner across sessions. No email, no password, no real name — just a username only you know.</p>
        </div>""", unsafe_allow_html=True)
        c1, c2 = st.columns([4, 1])
        with c1:
            uname = st.text_input("Username", placeholder="e.g. bluefox42", key="profile_input", label_visibility="collapsed")
        with c2:
            if st.button("Sign in", key="profile_signin", use_container_width=True):
                if uname and uname.strip():
                    st.session_state.user_profile = uname.strip()
                    data = load_user_data(uname.strip())
                    if data:
                        if "tracker" in data: st.session_state.tracker_history = data["tracker"]
                        if "sleep" in data: st.session_state.sleep_log = data["sleep"]
                        if "letter" in data: st.session_state.letter = data["letter"]
                        if "mir_plan" in data: st.session_state.mir_plan = data["mir_plan"]
                    st.rerun()

def page_student_hub():
    render_top_bar()
    render_breadcrumb(("MedMind", "gateway"), "Student hub")

    ctx = get_personalization_context()
    is_onboarding = render_onboarding_or_nudge(ctx)
    if is_onboarding and not st.session_state.get("onboarding_skipped", False):
        render_footer()
        return

    st.markdown("""
    <div style="padding:24px 0 16px;">
        <h2 style="font-family:'Fraunces',serif;font-size:30px;font-weight:600;color:#0F2B3D;margin-bottom:8px;">Wellbeing centre</h2>
        <p style="font-size:16px;color:#5a6a7e;line-height:1.65;">Tools designed around the rhythms of medical training.</p>
    </div>""", unsafe_allow_html=True)

    # Adaptive top recommendation — only if student has taken the screening
    if ctx["has_screening"]:
        top = ctx["top_risk"]
        level = ctx["top_risk_level"]
        reco_map = {
            "burnout": ("Burnout exhaustion tools", "exam_mode", "Exam period mode — quick protective actions for high-pressure weeks", "#FAEEDA", "#FAC775", "#633806", "#854F0B"),
            "depression": ("Connection and reflection", "stories", "Peer stories from students who have been there", "#FBEAF0", "#F4C0D1", "#72243E", "#993556"),
            "anxiety": ("Grounding and pacing", "coping", "Coping strategies — breathing, grounding, study scheduling", "#EEEDFE", "#CECBF6", "#3C3489", "#534AB7"),
        }
        if top in reco_map:
            title, page, desc, bg, border, hcolor, pcolor = reco_map[top]
            st.markdown(f"""<div style="background:{bg};border:1px solid {border};border-radius:14px;padding:20px 22px;margin-bottom:16px;">
                <span style="font-size:11px;font-weight:600;color:{hcolor};text-transform:uppercase;letter-spacing:0.5px;">Recommended for you</span>
                <h4 style="font-family:'Fraunces',serif;font-size:17px;font-weight:500;color:{hcolor};margin:4px 0 4px;">{title}</h4>
                <p style="font-size:14px;color:{pcolor};line-height:1.55;margin-bottom:12px;">{desc}</p>
            </div>""", unsafe_allow_html=True)
            if st.button(f"Open {title.lower()}", key=f"adapt_{top}", use_container_width=True):
                st.session_state.page = page; st.rerun()

    tabs = st.tabs(["Screening & assessment", "Support & resources", "Tracking & reflection", "MIR preparation"])

    # ═══ TAB 1: SCREENING ═══
    with tabs[0]:
        st.markdown("""<div style="background:#D6EDF2;border:1px solid #48A8C9;border-radius:16px;padding:28px 24px;margin-top:12px;">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#0F2B3D" stroke-width="1.5" style="margin-bottom:12px;"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
            <h3 style="font-family:'Fraunces',serif;font-size:18px;font-weight:500;color:#0F2B3D;margin-bottom:6px;">Take the screening</h3>
            <p style="font-size:15px;color:#48A8C9;line-height:1.6;">Five minutes, completely confidential. Your risk profile, explained with the factors that matter most.</p>
        </div>""", unsafe_allow_html=True)
        if st.button("Begin screening", key="s_screen", use_container_width=True):
            st.session_state.page = "screening"; st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""<div style="background:#D6EDF2;border:1px solid #48A8C9;border-radius:14px;padding:20px 18px;margin-top:12px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0F2B3D" stroke-width="1.5" style="margin-bottom:10px;"><path d="M18 20V10"/><path d="M12 20V4"/><path d="M6 20v-6"/></svg>
                <h4 style="font-family:'Fraunces',serif;font-size:16px;font-weight:500;color:#0F2B3D;margin-bottom:4px;">Am I normal?</h4>
                <p style="font-size:14px;color:#48A8C9;line-height:1.55;">Compare against 5,216 students nationally</p>
            </div>""", unsafe_allow_html=True)
            if st.button("Compare my profile", key="s_benchmark", use_container_width=True):
                st.session_state.page = "benchmark"; st.rerun()
        with c2:
            st.markdown("""<div style="background:#FAEEDA;border:1px solid #FAC775;border-radius:14px;padding:20px 18px;margin-top:12px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#633806" stroke-width="1.5" style="margin-bottom:10px;"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
                <h4 style="font-family:'Fraunces',serif;font-size:16px;font-weight:500;color:#633806;margin-bottom:4px;">Exam period mode</h4>
                <p style="font-size:14px;color:#854F0B;line-height:1.55;">Quick tools for high-pressure weeks</p>
            </div>""", unsafe_allow_html=True)
            if st.button("Activate", key="s_exam", use_container_width=True):
                st.session_state.page = "exam_mode"; st.rerun()

    # ═══ TAB 2: SUPPORT ═══
    with tabs[1]:
        # Featured MindGuide card
        st.markdown(
            '<div style="background:#fff;border:1px solid #E5DCC5;border-radius:16px;padding:24px 26px;margin-top:12px;margin-bottom:16px;position:relative;overflow:hidden;">'
            '<div style="position:absolute;top:0;right:0;width:160px;height:160px;border-radius:50%;background:#D6EDF2;opacity:0.35;transform:translate(35%,-35%);pointer-events:none;"></div>'
            '<div style="display:flex;align-items:center;gap:18px;position:relative;z-index:2;">'
            '<div style="width:60px;height:60px;border-radius:16px;background:#D6EDF2;display:flex;align-items:center;justify-content:center;flex-shrink:0;">'
            '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#0F2B3D" stroke-width="1.7"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>'
            '</div>'
            '<div style="flex:1;">'
            '<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">'
            '<h3 style="font-family:\'Fraunces\',serif;font-size:20px;font-weight:500;color:#0F2B3D;margin:0;letter-spacing:-0.3px;">Talk to MindGuide</h3>'
            '<span style="font-size:10px;font-weight:600;color:#48A8C9;background:#D6EDF2;padding:3px 10px;border-radius:100px;text-transform:uppercase;letter-spacing:0.5px;">AI companion</span>'
            '</div>'
            '<p style="font-family:\'Inter\',sans-serif;font-size:14px;color:#6C766F;line-height:1.55;margin:0;">A supportive space to reflect on how you\'re doing, think through a rough week, or figure out what kind of support might help. Not therapy — for ongoing issues, speak with a professional.</p>'
            '</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )
        if st.button("Open MindGuide", key="s_mindguide", use_container_width=True):
            st.session_state.page = "agent"; st.rerun()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""<div style="background:#E1F5EE;border:1px solid #9FE1CB;border-radius:14px;padding:20px 18px;margin-top:12px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#085041" stroke-width="1.5" style="margin-bottom:10px;"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
                <h4 style="font-family:'Fraunces',serif;font-size:16px;font-weight:500;color:#085041;margin-bottom:4px;">Find resources</h4>
                <p style="font-size:14px;color:#0F6E56;line-height:1.55;">37 universities across Spain</p>
            </div>""", unsafe_allow_html=True)
            if st.button("Find resources", key="s_resources", use_container_width=True):
                st.session_state.page = "resources"; st.rerun()
        with c2:
            st.markdown("""<div style="background:#EEEDFE;border:1px solid #CECBF6;border-radius:14px;padding:20px 18px;margin-top:12px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3C3489" stroke-width="1.5" style="margin-bottom:10px;"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
                <h4 style="font-family:'Fraunces',serif;font-size:16px;font-weight:500;color:#3C3489;margin-bottom:4px;">Coping strategies</h4>
                <p style="font-size:14px;color:#534AB7;line-height:1.55;">Practical, not generic</p>
            </div>""", unsafe_allow_html=True)
            if st.button("Explore strategies", key="s_coping", use_container_width=True):
                st.session_state.page = "coping"; st.rerun()
        with c3:
            st.markdown("""<div style="background:#FBEAF0;border:1px solid #F4C0D1;border-radius:14px;padding:20px 18px;margin-top:12px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#72243E" stroke-width="1.5" style="margin-bottom:10px;"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                <h4 style="font-family:'Fraunces',serif;font-size:16px;font-weight:500;color:#72243E;margin-bottom:4px;">Peer stories</h4>
                <p style="font-size:14px;color:#993556;line-height:1.55;">From students who've been there</p>
            </div>""", unsafe_allow_html=True)
            if st.button("Read stories", key="s_stories", use_container_width=True):
                st.session_state.page = "stories"; st.rerun()

        # Crisis
        st.markdown("""
        <div style="margin-top:16px;padding:16px 20px;background:#FCEBEB;border:1px solid #F7C1C1;border-radius:14px;display:flex;align-items:center;gap:14px;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#791F1F" stroke-width="1.5" style="flex-shrink:0;"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72"/></svg>
            <div>
                <p style="font-size:13px;font-weight:500;color:#791F1F;margin:0 0 2px;">If you need help now</p>
                <p style="font-size:12px;color:#A32D2D;margin:0;"><strong>024</strong> — free, confidential, 24/7 · <a href="https://www.icomem.es/seccion/SALUD-MENTAL-MEDICO" target="_blank" style="color:#791F1F;text-decoration:underline;">PAIME</a></p>
            </div>
        </div>""", unsafe_allow_html=True)

    # ═══ TAB 3: TRACKING ═══
    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""<div style="background:#E1F5EE;border:1px solid #9FE1CB;border-radius:14px;padding:20px 18px;margin-top:12px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#085041" stroke-width="1.5" style="margin-bottom:10px;"><path d="M12 2a10 10 0 1 0 10 10"/><path d="M12 6v6l4 2"/></svg>
                <h4 style="font-family:'Fraunces',serif;font-size:16px;font-weight:500;color:#085041;margin-bottom:4px;">Monthly tracker</h4>
                <p style="font-size:14px;color:#0F6E56;line-height:1.55;">Spot patterns before they become problems</p>
            </div>""", unsafe_allow_html=True)
            if st.button("Open tracker", key="s_tracker", use_container_width=True):
                st.session_state.page = "tracker"; st.rerun()
        with c2:
            st.markdown("""<div style="background:#FBEAF0;border:1px solid #F4C0D1;border-radius:14px;padding:20px 18px;margin-top:12px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#72243E" stroke-width="1.5" style="margin-bottom:10px;"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
                <h4 style="font-family:'Fraunces',serif;font-size:16px;font-weight:500;color:#72243E;margin-bottom:4px;">Sleep and routine</h4>
                <p style="font-size:14px;color:#993556;line-height:1.55;">Three daily inputs, one clear picture</p>
            </div>""", unsafe_allow_html=True)
            if st.button("Track today", key="s_sleep", use_container_width=True):
                st.session_state.page = "sleep_tracker"; st.rerun()

        st.markdown("""<div style="background:#EEEDFE;border:1px solid #CECBF6;border-radius:14px;padding:20px 18px;margin-top:12px;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3C3489" stroke-width="1.5" style="margin-bottom:10px;"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/></svg>
            <h4 style="font-family:'Fraunces',serif;font-size:16px;font-weight:500;color:#3C3489;margin-bottom:4px;">Letter to your future self</h4>
            <p style="font-size:14px;color:#534AB7;line-height:1.55;">Write something kind on a good day. Read it when things feel impossible.</p>
        </div>""", unsafe_allow_html=True)
        if st.button("Write a letter", key="s_letter", use_container_width=True):
            st.session_state.page = "letter"; st.rerun()

    # ═══ TAB 4: MIR ═══
    with tabs[3]:
        st.markdown("""
        <div style="margin-top:12px;padding:22px 20px;background:#D6EDF2;border:1px solid #48A8C9;border-radius:14px;">
            <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0F2B3D" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>
                <p style="font-family:'Fraunces',serif;font-size:17px;font-weight:500;color:#0F2B3D;margin:0;">MIR preparation and wellbeing</p>
            </div>
            <p style="font-size:13px;color:#48A8C9;line-height:1.65;">Wellness roadmap, specialty alignment tool, preparation resources and academy links, weekly study planner, peer stories, and post-MIR transition guidance.</p>
        </div>""", unsafe_allow_html=True)
        if st.button("Open MIR hub", key="s_mir", use_container_width=True):
            st.session_state.page = "mir_student"; st.rerun()

    # Switch role
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("Switch to institutional view", key="switch_to_inst", use_container_width=True):
            st.session_state.role = "institution"; st.session_state.page = "institution_hub"; st.rerun()

    render_footer()


# ============================================================================
# PAGE: INSTITUTION HUB
# ============================================================================

def page_institution_hub():
    render_top_bar()
    render_breadcrumb(("MedMind", "gateway"), "Institution hub")

    st.markdown("""
    <div style="padding:32px 0 16px;">
        <h2 style="font-family:'Fraunces',serif;font-size:30px;font-weight:600;color:#0F2B3D;margin-bottom:8px;">Institutional intelligence</h2>
        <p style="font-size:16px;color:#5a6a7e;line-height:1.65;">Population-level insights, intervention planning, and implementation tools.</p>
    </div>""", unsafe_allow_html=True)

    tabs = st.tabs(["Data & analysis", "Interventions & tools", "Implementation", "MIR support"])

    # ═══ TAB 1: DATA ═══
    with tabs[0]:
        st.markdown("""<div style="background:#D6EDF2;border:1px solid #48A8C9;border-radius:16px;padding:28px 24px;margin-top:12px;">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#0F2B3D" stroke-width="1.5" style="margin-bottom:12px;"><path d="M18 20V10"/><path d="M12 20V4"/><path d="M6 20v-6"/></svg>
            <h3 style="font-family:'Fraunces',serif;font-size:18px;font-weight:500;color:#0F2B3D;margin-bottom:6px;">DABE 2020 dashboard</h3>
            <p style="font-size:15px;color:#48A8C9;line-height:1.6;">Population-level data across 5,216 students: prevalence by year, gender breakdowns, comorbidity patterns, and SHAP factor rankings.</p>
        </div>""", unsafe_allow_html=True)
        if st.button("Open dashboard", key="i_dash", use_container_width=True):
            st.session_state.page = "dashboard"; st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""<div style="background:#FAEEDA;border:1px solid #FAC775;border-radius:14px;padding:20px 18px;margin-top:12px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#633806" stroke-width="1.5" style="margin-bottom:10px;"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
                <h4 style="font-family:'Fraunces',serif;font-size:16px;font-weight:500;color:#633806;margin-bottom:4px;">Historical trends</h4>
                <p style="font-size:14px;color:#854F0B;line-height:1.55;">Pre-pandemic vs post-COVID comparison</p>
            </div>""", unsafe_allow_html=True)
            if st.button("View trends", key="i_hist", use_container_width=True):
                st.session_state.page = "historical"; st.rerun()
        with c2:
            st.markdown("""<div class="mod-card">
                <div class="mc-icon" style="background:#fae8ee;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#72243E" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/><path d="M9 3v18"/><path d="M15 3v18"/></svg></div>
                <h3>Risk factor heatmap</h3>
                <p>Which risk factors peak in which year, filterable by gender.</p>
            </div>""", unsafe_allow_html=True)
            if st.button("View heatmap", key="i_heatmap", use_container_width=True):
                st.session_state.page = "heatmap"; st.rerun()

        # Featured: Explainable risk factors (SHAP)
        st.markdown(
            '<div style="background:#fff;border:1px solid #E5DCC5;border-radius:16px;padding:24px 26px;margin-top:16px;position:relative;overflow:hidden;">'
            '<div style="position:absolute;top:0;right:0;width:160px;height:160px;border-radius:50%;background:#D6EDF2;opacity:0.35;transform:translate(35%,-35%);pointer-events:none;"></div>'
            '<div style="display:flex;align-items:center;gap:18px;position:relative;z-index:2;">'
            '<div style="width:60px;height:60px;border-radius:16px;background:#D6EDF2;display:flex;align-items:center;justify-content:center;flex-shrink:0;">'
            '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#0F2B3D" stroke-width="1.7"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
            '</div>'
            '<div style="flex:1;">'
            '<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">'
            '<h3 style="font-family:\'Fraunces\',serif;font-size:20px;font-weight:500;color:#0F2B3D;margin:0;letter-spacing:-0.3px;">Explainable risk factors</h3>'
            '<span style="font-size:10px;font-weight:600;color:#48A8C9;background:#D6EDF2;padding:3px 10px;border-radius:100px;text-transform:uppercase;letter-spacing:0.5px;">SHAP rankings</span>'
            '</div>'
            '<p style="font-family:\'Inter\',sans-serif;font-size:14px;color:#6C766F;line-height:1.55;margin:0;">What actually drives each prediction. Top 5 features per outcome with direction, effect size, and clinical interpretation — plus the cross-cutting factors that matter for multiple conditions.</p>'
            '</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )
        if st.button("View factor rankings", key="i_shap", use_container_width=True):
            st.session_state.page = "shap_rankings"; st.rerun()

        st.markdown("""<div class="mod-card">
            <div class="mc-icon" style="background:#eee9f5;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3C3489" stroke-width="2"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg></div>
            <h3>Simulated reporting dashboard</h3>
            <p>Realistic synthetic data showing what a semester of screening looks like in practice — participation rates, risk tiers, referral funnels.</p>
        </div>""", unsafe_allow_html=True)
        if st.button("Explore simulation", key="i_sim", use_container_width=True):
            st.session_state.page = "sim_dashboard"; st.rerun()

    # ═══ TAB 2: INTERVENTIONS ═══
    with tabs[1]:
        # Featured: Intervention recommender
        st.markdown("""<div style="background:#D6EDF2;border:1px solid #48A8C9;border-radius:16px;padding:28px 24px;margin-top:12px;">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#0F2B3D" stroke-width="1.5" style="margin-bottom:12px;"><path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="8.5" cy="7" r="4"/><line x1="20" y1="8" x2="20" y2="14"/><line x1="23" y1="11" x2="17" y2="11"/></svg>
            <h3 style="font-family:'Fraunces',serif;font-size:18px;font-weight:500;color:#0F2B3D;margin-bottom:6px;">What should we do first?</h3>
            <p style="font-size:15px;color:#48A8C9;line-height:1.6;">Input your university's data. Get a ranked, costed action plan with ROI estimates — backed by SHAP analysis and DABE benchmarks. This is where data becomes decisions.</p>
        </div>""", unsafe_allow_html=True)
        if st.button("Get your action plan", key="i_recommend", use_container_width=True):
            st.session_state.page = "intervention_recommender"; st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""<div class="mod-card" style="margin-top:12px;">
                <div class="mc-icon" style="background:#e1f5ee;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#085041" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M16 13H8"/><path d="M16 17H8"/></svg></div>
                <h3>Intervention playbook</h3>
                <p>Actions matched to each SHAP-identified risk factor, with citations.</p>
            </div>""", unsafe_allow_html=True)
            if st.button("Open playbook", key="i_play", use_container_width=True):
                st.session_state.page = "playbook"; st.rerun()
        with c2:
            st.markdown("""<div class="mod-card" style="margin-top:12px;">
                <div class="mc-icon" style="background:#eee9f5;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3C3489" stroke-width="2"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg></div>
                <h3>Cost-of-inaction calculator</h3>
                <p>Untreated mental health costs vs screening investment. The business case.</p>
            </div>""", unsafe_allow_html=True)
            if st.button("Open calculator", key="i_calc", use_container_width=True):
                st.session_state.page = "calculator"; st.rerun()

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("""<div class="mod-card">
                <div class="mc-icon" style="background:#e4edf7;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#48A8C9" stroke-width="2"><path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" y1="22" x2="4" y2="15"/></svg></div>
                <h3>Benchmark your university</h3>
                <p>Input your data, compare against DABE national averages.</p>
            </div>""", unsafe_allow_html=True)
            if st.button("Compare data", key="i_bench", use_container_width=True):
                st.session_state.page = "inst_benchmark"; st.rerun()
        with c4:
            st.markdown("""<div class="mod-card">
                <div class="mc-icon" style="background:#e4edf7;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#48A8C9" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/></svg></div>
                <h3>Policy brief generator</h3>
                <p>One-page downloadable summary for your dean or board.</p>
            </div>""", unsafe_allow_html=True)
            if st.button("Generate brief", key="i_policy", use_container_width=True):
                st.session_state.page = "policy_brief"; st.rerun()

    # ═══ TAB 3: IMPLEMENTATION ═══
    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""<div style="background:#D6EDF2;border:1px solid #48A8C9;border-radius:14px;padding:20px 18px;margin-top:12px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0F2B3D" stroke-width="1.5" style="margin-bottom:10px;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                <h4 style="font-family:'Fraunces',serif;font-size:16px;font-weight:500;color:#0F2B3D;margin-bottom:4px;">Upload screening data</h4>
                <p style="font-size:14px;color:#48A8C9;line-height:1.55;">CSV upload to generate your own dashboard</p>
            </div>""", unsafe_allow_html=True)
            if st.button("Upload CSV", key="i_csv", use_container_width=True):
                st.session_state.page = "csv_upload"; st.rerun()
        with c2:
            st.markdown("""<div style="background:#FCEBEB;border:1px solid #F7C1C1;border-radius:14px;padding:20px 18px;margin-top:12px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#791F1F" stroke-width="1.5" style="margin-bottom:10px;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                <h4 style="font-family:'Fraunces',serif;font-size:16px;font-weight:500;color:#791F1F;margin-bottom:4px;">Wellbeing triage dashboard</h4>
                <p style="font-size:14px;color:#A32D2D;line-height:1.55;">Red-flag workflow for wellbeing officers</p>
            </div>""", unsafe_allow_html=True)
            if st.button("Open triage dashboard", key="i_triage", use_container_width=True):
                st.session_state.page = "triage_dashboard"; st.rerun()

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("""<div class="mod-card">
                <div class="mc-icon" style="background:#fdf4e7;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#854f0b" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg></div>
                <h3>Early warning system guide</h3>
                <p>Frequency, questionnaires, referral protocols, communications, privacy, and integration.</p>
            </div>""", unsafe_allow_html=True)
            if st.button("Read guide", key="i_ew", use_container_width=True):
                st.session_state.page = "earlywarning"; st.rerun()
        with c4:
            st.markdown("""<div class="mod-card">
                <div class="mc-icon" style="background:#e1f5ee;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#085041" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg></div>
                <h3>Programme evaluation</h3>
                <p>KPIs, measurement approach, and realistic timeline.</p>
            </div>""", unsafe_allow_html=True)
            if st.button("View framework", key="i_eval", use_container_width=True):
                st.session_state.page = "evaluation"; st.rerun()

        c5, c6 = st.columns(2)
        with c5:
            st.markdown("""<div class="mod-card">
                <div class="mc-icon" style="background:#e1f5ee;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#085041" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg></div>
                <h3>Resource directory</h3>
                <p>37 universities across Spain.</p>
            </div>""", unsafe_allow_html=True)
            if st.button("Browse resources", key="i_resources", use_container_width=True):
                st.session_state.page = "resources"; st.rerun()
        with c6:
            st.markdown("""<div style="background:#E1F5EE;border:1px solid #9FE1CB;border-radius:14px;padding:20px 18px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#085041" stroke-width="1.5" style="margin-bottom:10px;"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M12 18v-6"/><path d="M9 15h6"/></svg>
                <h4 style="font-family:'Fraunces',serif;font-size:16px;font-weight:500;color:#085041;margin-bottom:4px;">Data collection framework</h4>
                <p style="font-size:14px;color:#0F6E56;line-height:1.55;">Standardised protocol for contributing to model validation</p>
            </div>""", unsafe_allow_html=True)
            if st.button("View framework", key="i_datacollect", use_container_width=True):
                st.session_state.page = "data_collection"; st.rerun()

    # ═══ TAB 4: MIR ═══
    with tabs[3]:
        st.markdown("""
        <div style="margin-top:12px;padding:24px;background:#fff;border:1px solid #E5DCC5;border-radius:10px;border-left:3px solid #0F2B3D;">
            <div style="display:flex;align-items:center;gap:14px;margin-bottom:12px;">
                <div style="width:40px;height:40px;border-radius:10px;background:#e0e5ee;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0F2B3D" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>
                </div>
                <p style="font-family:'Fraunces',serif;font-size:17px;font-weight:500;color:#0F2B3D;margin:0;">MIR preparation support protocol</p>
            </div>
            <p style="font-size:14px;color:#3a4a5e;line-height:1.65;">Framework for supporting students through the highest-pressure period in Spanish medical training. Includes stress impact evidence, 8 institutional recommendations, and a 5-checkpoint cohort tracking framework with estimated stress trajectory data.</p>
        </div>""", unsafe_allow_html=True)
        if st.button("View MIR protocol", key="i_mir", use_container_width=True):
            st.session_state.page = "mir_institution"; st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("Switch to student view", key="switch_to_stu", use_container_width=True):
            st.session_state.role = "student"; st.session_state.page = "student_hub"; st.rerun()

    render_footer()


# ============================================================================
# PAGE: SCREENING
# ============================================================================

def page_screening():
    render_top_bar()
    render_breadcrumb(("MedMind", "gateway"), ("Student hub", "student_hub"), "Screening")
    section_idx = st.session_state.get("section", 0)
    total = len(SCREENING_SECTIONS)
    progress = (section_idx + 1) / total
    st.markdown(f'<div style="padding:12px 0 0;"><div style="height:4px;background:#E5DCC5;border-radius:4px;overflow:hidden;"><div style="height:100%;width:{progress*100:.0f}%;background:linear-gradient(90deg,#0F2B3D,#48a8c9);border-radius:4px;"></div></div><div style="display:flex;justify-content:space-between;margin-top:6px;"><span style="font-size:11px;color:#8c96a4;">Section {section_idx+1}/{total} — {SCREENING_SECTIONS[section_idx]["title"]}</span><span style="font-size:11px;color:#8c96a4;">{progress*100:.0f}%</span></div></div>', unsafe_allow_html=True)
    section = SCREENING_SECTIONS[section_idx]
    st.markdown(f'<div style="padding:16px 0 8px;"><h3 style="font-size:18px;font-weight:600;color:#0F2B3D;margin-bottom:4px;">{section["title"]}</h3><p style="font-size:13px;color:#6b7a8d;">{section["subtitle"]}</p></div>', unsafe_allow_html=True)
    if "answers" not in st.session_state: st.session_state.answers = {}
    for q in section["questions"]:
        if q["type"] == "number":
            st.session_state.answers[q["id"]] = st.number_input(q["text"], q["min"], q["max"], st.session_state.answers.get(q["id"], q["default"]), key=f"q_{q['id']}")
        elif q["type"] == "select":
            opts = list(q["options"].keys())
            cur = st.session_state.answers.get(q["id"])
            di = 0
            if cur is not None:
                for i, (k, v) in enumerate(q["options"].items()):
                    if v == cur: di = i; break
            st.session_state.answers[q["id"]] = q["options"][st.selectbox(q["text"], opts, di, key=f"q_{q['id']}")]
        elif q["type"] == "slider":
            st.session_state.answers[q["id"]] = st.slider(q["text"], q["min"], q["max"], st.session_state.answers.get(q["id"], q["default"]), key=f"q_{q['id']}")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if section_idx > 0 and st.button("← Back", key="back"):
            st.session_state.section = section_idx - 1; st.rerun()
    with c3:
        if section_idx < total - 1:
            if st.button("Continue →", key="next"): st.session_state.section = section_idx + 1; st.rerun()
        else:
            if st.button("Get results →", key="submit"):
                a = st.session_state.answers
                a["n_problems"] = sum(a.get(k, 0) for k in a if k.startswith("prob_"))
                for c in ['pharma_anxiolytics','pharma_mood_stabilizers','pharma_antidepressants','pharma_antipsychotics','pharma_other']:
                    if c not in a: a[c] = 0
                st.session_state.page = "results"; st.rerun()


# ============================================================================
# PAGE: RESULTS
# ============================================================================

def page_results():
    render_top_bar()
    render_breadcrumb(("MedMind", "gateway"), ("Student hub", "student_hub"), "Your results")
    models, features = load_models()
    answers = st.session_state.answers

    targets = {'depression': 'Depression', 'burnout': 'Burnout', 'anxiety': 'Anxiety', 'suicidal_ideation': '_si'}
    results = {}
    for tk, tl in targets.items():
        mk = f"{tk}_B"
        if mk in models and mk in features:
            try:
                X = pd.DataFrame([{f: answers.get(f, 0) for f in features[mk]}])
                results[tk] = models[mk].predict_proba(X)[:, 1][0]
            except: results[tk] = 0.0
    st.session_state.screening_results = results

    if "pages_visited" not in st.session_state: st.session_state.pages_visited = []
    if "results" not in st.session_state.pages_visited: st.session_state.pages_visited.append("results")

    vis = {k: v for k, v in results.items() if k != 'suicidal_ideation'}
    ht = max(vis, key=vis.get) if vis else 'depression'
    hp = vis.get(ht, 0)
    hl = {'depression':'Depression','burnout':'Burnout','anxiety':'Anxiety'}.get(ht, 'Depression')
    level = get_risk_level(hp)[0].lower()
    year = answers.get("course_year", 3)
    support = answers.get("social_support_n", 3)
    satisfaction = answers.get("acad_satisfaction_choice", 3)
    n_events = sum(answers.get(f"eve_{i}", 0) for i in range(1, 6))

    flabels = {'social_support_n':'Social support network','social_support_q2':'Emotional support quality','social_support_q3':'Practical support','social_support_q4':'Someone to talk to','n_problems':'Number of current problems','acad_satisfaction_studies':'Study satisfaction','acad_satisfaction_choice':'Career choice satisfaction','acad_effort_match':'Effort-grade alignment','course_year':'Course year','Age':'Age','attendance':'Class attendance','takes_psychopharm':'Psychiatric medication','prob_academic_performance':'Academic worries','prob_health':'Health concerns','prob_time_management':'Time management','prob_family':'Family difficulties','eve_1':'Personal illness','eve_2':'Family illness','eve_3':'Financial difficulties','eve_4':'Relationship breakup','eve_5':'Bereavement','alcohol':'Alcohol','cannabis':'Cannabis','tobacco':'Tobacco'}

    # ═══ SECTION 1: Risk profile ═══
    st.markdown("""<div style="padding:24px 0 16px;">
        <h2 style="font-family:'Fraunces',serif;font-size:24px;font-weight:600;color:#0F2B3D;margin-bottom:6px;">Your screening results</h2>
        <p style="font-size:14px;color:#888780;">This is not a diagnosis. Here is your personalised risk profile.</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, (k, l) in zip([c1, c2, c3], [('depression','Depression'),('burnout','Burnout'),('anxiety','Anxiety')]):
        with col: st.markdown(render_risk_card(l, results.get(k, 0)), unsafe_allow_html=True)

    top_idx, importances, feat_list = [], np.array([]), []
    mk = f"{ht}_B"
    if mk in models and mk in features:
        feat_list = features[mk]
        X = pd.DataFrame([{f: answers.get(f, 0) for f in feat_list}])
        clf = models[mk].named_steps.get('clf', models[mk])
        importances = clf.feature_importances_ if hasattr(clf, 'feature_importances_') else np.abs(clf.coef_[0]) if hasattr(clf, 'coef_') else np.ones(len(feat_list))/len(feat_list)
        top_idx = np.argsort(importances)[-5:][::-1]
        st.markdown(f'<div class="factor-box"><h4>Top factors contributing to your {hl.lower()} risk</h4>', unsafe_allow_html=True)
        mx = importances[top_idx[0]] if importances[top_idx[0]] > 0 else 1
        for idx in top_idx:
            st.markdown(render_factor_bar(flabels.get(feat_list[idx], feat_list[idx].replace('_',' ').title()), importances[idx]/mx*0.2, 0.25), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Soft crisis protocol
    if results.get('suicidal_ideation', 0) >= 0.10:
        st.markdown('<div class="crisis-box"><h4>We want to make sure you are supported</h4><p>Confidential, professional support is available right now. You do not have to manage this alone.</p><a class="hotline" href="tel:024">Call 024</a><p style="margin-top:10px;font-size:12px;color:#5a6a7e;"><a href="https://www.icomem.es/seccion/SALUD-MENTAL-MEDICO" target="_blank" style="color:#5a6a7e;text-decoration:underline;">PAIME</a> for medical students. 24/7.</p></div>', unsafe_allow_html=True)

    # AI guidance
    rd = {"highest_target": ht, "highest_level": level, "probability": round(hp, 2), "top_factors": [{"name": feat_list[i], "importance": round(float(importances[i]), 4)} for i in top_idx[:3]] if len(top_idx) else [], "course_year": year, "gender": "female" if answers.get("gender_female", 0) == 1 else "male"}
    g = generate_llm_response(rd) or get_fallback_guidance(rd)
    st.markdown(f'<div class="ai-box"><div class="ai-icon">AI</div><div class="ai-text"><strong>Personalised guidance:</strong> {g}</div></div>', unsafe_allow_html=True)

    # ═══ SECTION 2: Action plan (inline) ═══
    st.markdown('<p class="section-header">Your next steps</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:14px;color:#888780;margin-bottom:12px;">Tailored to your year, risk profile, and circumstances.</p>', unsafe_allow_html=True)

    steps = []
    ht_label = ht.replace('_', ' ')
    if level == "high":
        steps.append(("This week", "Book a counselling appointment", f"Your {ht_label} indicators are elevated. PAIME (024) can often see you faster than university services."))
    elif level == "moderate":
        steps.append(("This week", "Take one concrete step", f"Moderate {ht_label} indicators. Tell a trusted person, try a technique below, or look up counselling."))
    else:
        steps.append(("This week", "Protect what is working", "Your profile is relatively stable. Maintain social connections, sleep, and study boundaries."))

    if support <= 2:
        steps.append(("This month", "Strengthen your support", "Your social support is low. One action: reach out to one person this week."))
    elif satisfaction <= 2:
        steps.append(("This month", "Reconnect with your motivation", f"As a Year {year} student, consider career counselling or values clarification."))
    else:
        steps.append(("This month", "Set up self-monitoring", "Start tracking monthly. A 2-minute check-in helps spot patterns early."))

    if year >= 4:
        steps.append(("Ongoing", "Navigate clinical training", f"Year {year} is intensive. Buddy up, set recovery boundaries, use exam mode."))
    elif n_events >= 2:
        steps.append(("Ongoing", "Get support for life stressors", f"You are managing {n_events} life events. Consider requesting mitigating circumstances."))
    else:
        steps.append(("Ongoing", "Build sustainable habits", "Exercise, consistent sleep, one non-medical activity per week."))

    clrs = ["#48A8C9", "#0F6E56", "#534AB7"]
    for i, (timing, title, desc) in enumerate(steps):
        c = clrs[i % 3]
        st.markdown(f"""<div style="display:flex;gap:16px;margin-bottom:14px;">
            <div style="min-width:4px;background:{c};border-radius:2px;"></div>
            <div><span style="font-size:11px;font-weight:600;color:{c};text-transform:uppercase;letter-spacing:0.5px;">{timing}</span>
            <h4 style="font-family:'Fraunces',serif;font-size:16px;font-weight:500;color:#0F2B3D;margin:4px 0 6px;">{title}</h4>
            <p style="font-size:14px;color:#3a4a5e;line-height:1.65;margin:0;">{desc}</p></div>
        </div>""", unsafe_allow_html=True)

    # ═══ SECTION 3: Relevant coping strategies ═══
    coping_map = {
        'burnout': ("exhaustion and burnout", [("Micro-recovery breaks", "5-minute breaks every 90 minutes. Step away, stretch, look out a window."), ("Sleep hygiene", "Consistent bedtime even during exams. No screens 30 min before."), ("Protected exercise", "30 min moderate exercise 3x/week. Effect size comparable to mild antidepressants.")]),
        'anxiety': ("anxiety and worry", [("4-7-8 breathing", "Inhale 4s, hold 7s, exhale 8s. Repeat 3x. Works anywhere, anytime."), ("5-4-3-2-1 grounding", "5 things you see, 4 touch, 3 hear, 2 smell, 1 taste."), ("Study scheduling", "Specific tasks per block, not vague goals. Concrete plans reduce anxiety.")]),
        'depression': ("low mood and depression", [("Social reconnection", "One message to one person today. One connection per week builds momentum."), ("Energy management", "Track energy across the day. Schedule demanding tasks at your peaks."), ("Structured activity", "One extracurricular per week. Automatic social contact without initiation pressure.")])
    }
    if ht in coping_map:
        cat, strats = coping_map[ht]
        st.markdown(f'<p class="section-header">Strategies for {cat}</p>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:14px;color:#888780;margin-bottom:12px;">Based on your highest risk area.</p>', unsafe_allow_html=True)
        for t, d in strats:
            st.markdown(f'<div class="content-card"><h4>{t}</h4><p>{d}</p></div>', unsafe_allow_html=True)
        if st.button("See all coping strategies", key="res_coping", use_container_width=True):
            st.session_state.page = "coping"; st.rerun()

    # ═══ SECTION 4: Resources ═══
    st.markdown('<p class="section-header">Support available to you</p>', unsafe_allow_html=True)
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.markdown('<div style="background:#FCEBEB;border:1px solid #F7C1C1;border-radius:14px;padding:20px;text-align:center;"><h4 style="font-size:15px;color:#791F1F;margin-bottom:4px;">024 Crisis line</h4><p style="font-size:13px;color:#A32D2D;">Free, confidential, 24/7</p></div>', unsafe_allow_html=True)
    with rc2:
        st.markdown('<div style="background:#D6EDF2;border:1px solid #48A8C9;border-radius:14px;padding:20px;text-align:center;"><h4 style="font-size:15px;color:#0F2B3D;margin-bottom:4px;">PAIME</h4><p style="font-size:13px;color:#48A8C9;">For medical students</p></div>', unsafe_allow_html=True)
    with rc3:
        st.markdown('<div style="background:#E1F5EE;border:1px solid #9FE1CB;border-radius:14px;padding:20px;text-align:center;"><h4 style="font-size:15px;color:#085041;margin-bottom:4px;">Find your university</h4><p style="font-size:13px;color:#0F6E56;">37 universities</p></div>', unsafe_allow_html=True)
    if st.button("Find resources at my university", key="res_resources", use_container_width=True):
        st.session_state.page = "resources"; st.rerun()

    # ═══ SECTION 5: MindGuide ═══
    st.markdown("""<div style="margin-top:20px;padding:24px;background:#fff;border:1px solid #E5DCC5;border-radius:14px;text-align:center;">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#534AB7" stroke-width="1.5" style="margin-bottom:8px;"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        <h4 style="font-family:'Fraunces',serif;font-size:16px;color:#0F2B3D;margin-bottom:6px;">Want to talk about how you are feeling?</h4>
        <p style="font-size:14px;color:#888780;max-width:460px;margin:0 auto;">MindGuide can help you reflect on your results and find the right next step.</p>
    </div>""", unsafe_allow_html=True)
    if st.button("Talk to MindGuide", key="btn_chat", use_container_width=True):
        st.session_state.page = "agent"; st.rerun()

    st.markdown('<div style="text-align:center;padding:20px 0 8px;"><p style="font-size:11px;color:#B4B2A9;">This tool provides risk estimates, not a diagnosis. No data stored.</p></div>', unsafe_allow_html=True)
    nav_back_to_hub()

# ============================================================================

def page_shap_rankings():
    render_top_bar()
    render_breadcrumb(("MedMind", "gateway"), ("Institution hub", "institution_hub"), "Explainable risk factors")

    st.markdown('<div style="padding:16px 0 8px;"><h3 style="font-family:\'Fraunces\',serif;font-size:24px;font-weight:600;color:#0F2B3D;letter-spacing:-0.4px;">Explainable risk factors</h3><p style="font-size:14px;color:#6b7a8d;margin-top:4px;">What drives each prediction, ranked by SHAP contribution.</p></div>', unsafe_allow_html=True)

    # Intro / interpretation
    st.markdown(
        '<div style="background:#fff;border:1px solid #E5DCC5;border-radius:14px;padding:20px 24px;margin-bottom:20px;">'
        '<p style="font-family:\'Inter\',sans-serif;font-size:14px;color:#4A5560;line-height:1.7;margin:0;">'
        'MedMind\'s risk models are not black boxes. <strong style="color:#0F2B3D;">SHAP</strong> (SHapley Additive exPlanations) is a game-theoretic method that quantifies how much each feature contributes to a model\'s prediction. '
        'For each outcome, the ranked features below are the ones the model relies on most when deciding whether a student is at elevated risk. '
        'Bars extending right in <span style="color:#C23A3A;font-weight:500;">red</span> are risk factors — they push the prediction toward "at risk". '
        'Bars extending left in <span style="color:#0F6E56;font-weight:500;">green</span> are protective factors — their presence lowers predicted risk.'
        '</p>'
        '</div>',
        unsafe_allow_html=True
    )

    SHAP_DATA = {
        'Depression': {
            'model_auc': 0.947,
            'features': [
                ('State anxiety', 0.22, 'r', 'Students with high baseline anxiety are most likely to meet clinical depression thresholds'),
                ('Emotional exhaustion', 0.15, 'r', 'Burnout exhaustion is the second strongest depression predictor — the two conditions share underlying mechanisms'),
                ('Number of reported problems', 0.10, 'r', 'Each additional problem area (academic, financial, relational) raises depression risk'),
                ('Social support quality', -0.09, 'p', 'High-quality close relationships are the single strongest protective factor'),
                ('Academic satisfaction', -0.07, 'p', 'Meaningful engagement with studies buffers against depression'),
            ],
        },
        'Burnout': {
            'model_auc': 0.765,
            'features': [
                ('State anxiety', 0.14, 'r', 'Anxious students burn out faster — chronic stress depletes emotional reserves'),
                ('Academic problems', 0.11, 'r', 'Accumulated academic difficulties directly drive exhaustion and cynicism'),
                ('Study satisfaction', -0.09, 'p', 'Finding meaning in the work is the strongest burnout buffer we measured'),
                ('Effort-grade alignment', -0.07, 'p', 'When effort is reflected in grades, burnout risk drops substantially'),
                ('Course year', 0.06, 'r', 'Burnout nearly doubles from Year 1 (22.6%) to Year 6 (44.8%) — clinical rotations are the main accelerant'),
            ],
        },
        'Anxiety': {
            'model_auc': 0.881,
            'features': [
                ('Emotional exhaustion', 0.18, 'r', 'Depleted students lose their capacity to regulate anxious thoughts'),
                ('Social support availability', -0.10, 'p', 'Having people to turn to is the strongest anxiety buffer'),
                ('Social support quality', -0.08, 'p', 'Relationship depth matters as much as relationship count'),
                ('Cynicism (burnout)', 0.07, 'r', 'Detachment from work feeds anxious rumination'),
                ('Academic satisfaction', -0.06, 'p', 'Meaning in the work reduces anxious symptoms'),
            ],
        },
        'Suicidal ideation': {
            'model_auc': 0.878,
            'features': [
                ('State anxiety', 0.16, 'r', 'High anxiety is the single strongest behavioural predictor of suicidal ideation'),
                ('Non-heterosexual orientation', 0.08, 'r', 'Minority stress pathway — an ethical priority for institutional support'),
                ('Current antidepressant use', 0.07, 'r', 'Likely reflects more severe baseline depression rather than medication effect'),
                ('Social support network size', -0.06, 'p', 'Having people who would notice absence is uniquely protective against SI'),
                ('Career satisfaction', -0.05, 'p', 'Doubt about the choice to study medicine is uniquely predictive for this outcome'),
            ],
        },
    }

    target = st.selectbox("Select outcome", list(SHAP_DATA.keys()), key="shap_target_main")
    data = SHAP_DATA[target]

    # Model summary header
    target_colors = {'Depression': '#C23A3A', 'Burnout': '#C78017', 'Anxiety': '#48A8C9', 'Suicidal ideation': '#7B4F9E'}
    tc = target_colors.get(target, '#0F2B3D')
    st.markdown(
        f'<div style="background:#fff;border:1px solid #E5DCC5;border-radius:14px;padding:18px 22px;margin:10px 0 18px;border-left:4px solid {tc};">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">'
        f'<div><span style="font-size:11px;font-weight:600;color:#888780;text-transform:uppercase;letter-spacing:0.5px;">Model performance</span>'
        f'<h4 style="font-family:\'Fraunces\',serif;font-size:18px;font-weight:500;color:#0F2B3D;margin:2px 0 0;">{target} model — AUC {data["model_auc"]:.3f}</h4></div>'
        f'<div style="font-size:12px;color:#5a6a7e;text-align:right;max-width:320px;">Trained on DABE 2020 (n=5,216). Features below are the top 5 contributors by mean absolute SHAP value.</div>'
        f'</div></div>',
        unsafe_allow_html=True
    )

    # Feature rankings
    for rank, (name, value, direction, interpretation) in enumerate(data['features'], 1):
        is_risk = direction == 'r'
        bar_color = '#C23A3A' if is_risk else '#0F6E56'
        label_color = '#C23A3A' if is_risk else '#0F6E56'
        direction_label = 'Risk factor' if is_risk else 'Protective factor'
        bar_width = abs(value) / 0.25 * 100
        sign = '+' if value > 0 else ''

        card_html = (
            f'<div style="background:#fff;border:1px solid #E5DCC5;border-radius:12px;padding:18px 22px;margin-bottom:10px;">'
            f'<div style="display:flex;align-items:center;gap:14px;margin-bottom:10px;">'
            f'<div style="width:28px;height:28px;border-radius:50%;background:{bar_color};display:flex;align-items:center;justify-content:center;font-family:\'Inter\',sans-serif;font-size:13px;font-weight:600;color:#fff;flex-shrink:0;">{rank}</div>'
            f'<div style="flex:1;"><h4 style="font-family:\'Fraunces\',serif;font-size:16px;font-weight:500;color:#0F2B3D;margin:0;letter-spacing:-0.2px;">{name}</h4>'
            f'<span style="font-size:11px;font-weight:600;color:{label_color};text-transform:uppercase;letter-spacing:0.5px;">{direction_label}</span></div>'
            f'<div style="font-family:\'Inter\',sans-serif;font-size:14px;font-weight:600;color:{label_color};">{sign}{value*100:.0f}%</div>'
            f'</div>'
            f'<div style="height:10px;background:#F5F0E4;border-radius:5px;overflow:hidden;margin-bottom:10px;">'
            f'<div style="height:100%;width:{bar_width:.0f}%;background:{bar_color};border-radius:5px;"></div>'
            f'</div>'
            f'<p style="font-family:\'Inter\',sans-serif;font-size:13px;color:#6C766F;line-height:1.6;margin:0;">{interpretation}</p>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

    # Cross-cutting insight
    st.markdown('<p class="section-header" style="margin-top:24px;">Cross-cutting factors</p>', unsafe_allow_html=True)
    st.markdown(
        '<div style="background:#fff;border:1px solid #E5DCC5;border-radius:14px;padding:22px 24px;">'
        '<p style="font-family:\'Inter\',sans-serif;font-size:14px;color:#4A5560;line-height:1.7;margin-bottom:14px;">'
        'Three features appear as top drivers across multiple outcomes, making them highest-leverage intervention targets:'
        '</p>'
        '<div style="display:flex;gap:14px;flex-wrap:wrap;">'
        '<div style="flex:1;min-width:220px;padding:16px 18px;background:#FCEBEB;border-radius:10px;border-left:3px solid #C23A3A;">'
        '<p style="font-size:11px;font-weight:600;color:#C23A3A;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Risk factor</p>'
        '<h5 style="font-family:\'Fraunces\',serif;font-size:15px;font-weight:500;color:#0F2B3D;margin-bottom:4px;">State anxiety</h5>'
        '<p style="font-size:12px;color:#6C766F;line-height:1.5;margin:0;">Top or near-top predictor across all 4 outcomes. Targeting anxiety through CBT-based group workshops has system-wide benefits.</p>'
        '</div>'
        '<div style="flex:1;min-width:220px;padding:16px 18px;background:#FCEBEB;border-radius:10px;border-left:3px solid #C23A3A;">'
        '<p style="font-size:11px;font-weight:600;color:#C23A3A;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Risk factor</p>'
        '<h5 style="font-family:\'Fraunces\',serif;font-size:15px;font-weight:500;color:#0F2B3D;margin-bottom:4px;">Emotional exhaustion</h5>'
        '<p style="font-size:12px;color:#6C766F;line-height:1.5;margin:0;">Strong driver of depression and anxiety, and the core symptom of burnout. Protected rest and workload interventions address all three.</p>'
        '</div>'
        '<div style="flex:1;min-width:220px;padding:16px 18px;background:#E6F4EF;border-radius:10px;border-left:3px solid #0F6E56;">'
        '<p style="font-size:11px;font-weight:600;color:#0F6E56;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Protective factor</p>'
        '<h5 style="font-family:\'Fraunces\',serif;font-size:15px;font-weight:500;color:#0F2B3D;margin-bottom:4px;">Social support</h5>'
        '<p style="font-size:12px;color:#6C766F;line-height:1.5;margin:0;">Appears in the top 5 for all 4 outcomes. Peer mentorship and structured cohort-building are the #1 leverage intervention.</p>'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # Call to action
    st.markdown("<br>", unsafe_allow_html=True)
    ca1, ca2 = st.columns(2)
    with ca1:
        if st.button("Get ranked interventions for these factors", key="shap_to_reco", use_container_width=True):
            st.session_state.page = "intervention_recommender"; st.rerun()
    with ca2:
        if st.button("View the intervention playbook", key="shap_to_pb", use_container_width=True):
            st.session_state.page = "playbook"; st.rerun()

    # Methodological footer
    st.markdown(
        '<div style="background:#F5F0E4;border:1px solid #E5DCC5;border-radius:10px;padding:14px 18px;margin-top:20px;">'
        '<p style="font-family:\'Inter\',sans-serif;font-size:11px;color:#888780;line-height:1.6;margin:0;">'
        '<strong>Method.</strong> SHAP values computed on XGBoost models trained on DABE 2020 (n=5,216, 43 Spanish medical schools). '
        'Values shown are mean absolute SHAP contributions on a standardised scale. Sign indicates direction of effect (positive = risk, negative = protective). '
        'Full feature set, model hyperparameters, and cross-validation results are documented in the thesis Methods chapter. '
        '<em>SHAP measures predictive contribution, not causation — interpret as "the model relies on this feature", not "this feature causes the outcome".</em>'
        '</p>'
        '</div>',
        unsafe_allow_html=True
    )

    nav_back_to_hub()


def page_dashboard():
    render_top_bar()
    render_breadcrumb(("MedMind", "gateway"), ("Institution hub", "institution_hub"), "Dashboard")
    df = load_dashboard_data()
    if df is None:
        st.error("Dashboard data not found. Ensure dabe_clean_full.csv is in the app directory.")
        nav_back_to_hub(); return

    st.markdown('<div style="padding:16px 0 8px;"><h3 style="font-size:20px;font-weight:600;color:#0F2B3D;">Institutional dashboard</h3><p style="font-size:13px;color:#6b7a8d;">Population-level insights from the DABE 2020 dataset (n=5,216).</p></div>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">Population overview</p>', unsafe_allow_html=True)
    for c, v, l in zip(st.columns(5), [f"{len(df):,}", f"{df['target_depression_clinical'].mean()*100:.1f}%", f"{df['target_burnout'].mean()*100:.1f}%", f"{df['target_anxiety_high'].mean()*100:.1f}%", f"{df['target_suicidal_ideation'].mean()*100:.1f}%"], ["Total students","Depression","Burnout","Anxiety","Suicidal ideation"]):
        with c: st.markdown(f'<div class="metric-card"><div class="m-value">{v}</div><div class="m-label">{l}</div></div>', unsafe_allow_html=True)

    import matplotlib.pyplot as plt; import matplotlib; matplotlib.use('Agg')
    tmap = {'target_depression_clinical':'Depression','target_burnout':'Burnout','target_anxiety_high':'Anxiety','target_suicidal_ideation':'Suicidal ideation'}
    cmap = {'Depression':'#e24b4a','Burnout':'#ef9f27','Anxiety':'#48a8c9','Suicidal ideation':'#534ab7'}

    st.markdown('<p class="section-header">Year-by-year prevalence</p>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4)); fig.patch.set_alpha(0); ax.set_facecolor('none')
    for cn, lb in tmap.items():
        y = df.groupby('course_year')[cn].mean()*100
        ax.plot(y.index, y.values, 'o-', label=lb, color=cmap[lb], linewidth=2, markersize=6)
    ax.set_xlabel('Course year', fontsize=11, color='#5a6a7e'); ax.set_ylabel('Prevalence (%)', fontsize=11, color='#5a6a7e')
    ax.set_xticks(range(1,7)); ax.legend(fontsize=9); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(colors='#5a6a7e'); fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<p class="section-header">Risk distribution</p>', unsafe_allow_html=True)
    sel = st.selectbox("Select target", list(tmap.values()), key="dist_t")
    tc = [k for k,v in tmap.items() if v==sel][0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5)); fig.patch.set_alpha(0)
    ax=axes[0]; ax.set_facecolor('none')
    for g, clr in [('Female','#FF9999'),('Male','#66B2FF')]:
        s=df[df['gender']==g]; p=s[tc].mean()*100; ax.barh(g,p,color=clr,height=0.5); ax.text(p+0.5,g,f'{p:.1f}%',va='center',fontsize=10)
    ax.set_title(f'{sel} by gender',fontsize=12,color='#0F2B3D'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax=axes[1]; ax.set_facecolor('none'); df['phase']=np.where(df['course_year']<=3,'Preclinical (1-3)','Clinical (4-6)')
    for ph, clr in [('Preclinical (1-3)','#48a8c9'),('Clinical (4-6)','#0F2B3D')]:
        s=df[df['phase']==ph]; p=s[tc].mean()*100; ax.barh(ph,p,color=clr,height=0.5); ax.text(p+0.5,ph,f'{p:.1f}%',va='center',fontsize=10)
    ax.set_title(f'{sel} by phase',fontsize=12,color='#0F2B3D'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div style="background:#D6EDF2;border:1px solid #48A8C9;border-radius:12px;padding:16px 20px;margin-top:16px;display:flex;align-items:center;gap:12px;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0F2B3D" stroke-width="1.8" style="flex-shrink:0;"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg><span style="font-size:13px;color:#0F2B3D;">Looking for which factors drive each prediction? See the dedicated <strong>Explainable risk factors</strong> page in the Data &amp; analysis tab.</span></div>', unsafe_allow_html=True)

    nav_back_to_hub()


# ============================================================================
# FEATURE 1: HISTORICAL COMPARISON
# ============================================================================

def page_historical():
    render_top_bar()
    render_breadcrumb(("MedMind","gateway"),("Institution hub","institution_hub"),"Historical trends")
    st.markdown('<div style="padding:16px 0 8px;"><h3 style="font-size:20px;font-weight:600;color:#0F2B3D;">Pre-pandemic vs post-COVID comparison</h3><p style="font-size:13px;color:#6b7a8d;">DABE 2020 prevalence alongside published post-COVID meta-analyses.</p></div>', unsafe_allow_html=True)

    st.markdown("""<table class="comp-table"><tr><th>Condition</th><th>DABE 2020</th><th>Post-COVID</th><th>Trend</th><th>Source</th></tr>
    <tr><td><strong>Depression</strong></td><td>23.4%</td><td>48.0%</td><td class="trend-up">▲ +24.6 pp</td><td>Lin &amp; Saragih 2024 (130 studies, 132k)</td></tr>
    <tr><td><strong>Anxiety</strong></td><td>22.8%</td><td>45.0%</td><td class="trend-up">▲ +22.2 pp</td><td>Lin &amp; Saragih 2024</td></tr>
    <tr><td><strong>Burnout</strong></td><td>36.8%</td><td>73.5%</td><td class="trend-up">▲ +36.7 pp</td><td>Hawsawi et al. 2025 (Year 1)</td></tr>
    <tr><td><strong>Suicidal ideation</strong></td><td>10.6%</td><td>—</td><td class="trend-same">Insufficient data</td><td>—</td></tr>
    <tr><td><strong>Any symptoms</strong></td><td>~55%</td><td>52.7%</td><td class="trend-same">≈ Comparable</td><td>Soler et al. 2025 (Spanish)</td></tr></table>""", unsafe_allow_html=True)

    import matplotlib.pyplot as plt; import matplotlib; matplotlib.use('Agg')
    conds=['Depression','Anxiety','Burnout']; pre=[23.4,22.8,36.8]; post=[48.0,45.0,73.5]
    fig,ax=plt.subplots(figsize=(10,4.5)); fig.patch.set_alpha(0); ax.set_facecolor('none')
    x=np.arange(3); w=0.32
    b1=ax.bar(x-w/2,pre,w,label='DABE 2020',color='#48a8c9',edgecolor='white',zorder=3)
    b2=ax.bar(x+w/2,post,w,label='Post-COVID',color='#e24b4a',edgecolor='white',zorder=3)
    for b in b1: ax.text(b.get_x()+b.get_width()/2.,b.get_height()+1,f'{b.get_height():.1f}%',ha='center',fontsize=11,color='#0F2B3D',fontweight='500')
    for b in b2: ax.text(b.get_x()+b.get_width()/2.,b.get_height()+1,f'{b.get_height():.1f}%',ha='center',fontsize=11,color='#a32d2d',fontweight='500')
    ax.set_xticks(x);ax.set_xticklabels(conds,fontsize=12);ax.set_ylabel('Prevalence (%)',fontsize=11,color='#5a6a7e');ax.set_ylim(0,85)
    ax.legend(fontsize=10);ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False);ax.grid(axis='y',alpha=0.15)
    fig.tight_layout();st.pyplot(fig);plt.close()

    c1,c2=st.columns(2)
    with c1: st.markdown('<div class="content-card"><h4>🇪🇸 Spanish context</h4><p><strong>Soler et al. (2025):</strong> 52.7% of Spanish university students with clinical/subclinical symptoms. <strong>Ministry of Universities (2023):</strong> >50% need psychological support. <strong>Spain Mental Health Plan 2025-2027</strong> identifies students as a priority group.</p></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="content-card"><h4>🌍 International context</h4><p><strong>Lin &amp; Saragih (2024):</strong> 130 studies, 132k students — depression 48%, anxiety 45% during COVID. <strong>Hawsawi et al. (2025):</strong> 73.5% burnout, 44.2% depression in Year 1. <strong>MDPI 2025:</strong> narrative review of Spanish data 2010-2024.</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="content-card" style="border-left:4px solid #ef9f27;"><h4>⚠️ Interpretation note</h4><p>Post-COVID figures use different instruments (PHQ-9, GAD-7 vs BDI-II, STAI). Comparisons are directional, not exact.</p></div>', unsafe_allow_html=True)
    nav_back_to_hub()


# ============================================================================
# FEATURE 2: INTERVENTION PLAYBOOK
# ============================================================================

def page_playbook():
    render_top_bar()
    render_breadcrumb(("MedMind","gateway"),("Institution hub","institution_hub"),"Intervention playbook")
    st.markdown('<div style="padding:16px 0 8px;"><h3 style="font-size:20px;font-weight:600;color:#0F2B3D;">Intervention playbook</h3><p style="font-size:13px;color:#6b7a8d;">Evidence-based institutional actions matched to SHAP-identified risk factors.</p></div>', unsafe_allow_html=True)

    PB = [
        ("👥","Low social support","Depression, Anxiety, SI",[("Peer mentorship","Pair seniors with juniors. Monthly check-ins reduce isolation (Dyrbye et al., 2019)."),("Buddy systems","Rotation partners — no student enters clinical environments alone."),("Facilitated study groups","Weekly faculty-supported groups combining academic and social benefit."),("Social prescribing","Embed activity recommendations: sports, volunteering, societies.")]),
        ("🔥","High exhaustion","Burnout, Depression, Anxiety",[("Protected rest days","One full day/week with no obligations. 15-20% lower burnout in institutions with rest policies."),("Workload audits","Annual review of contact hours vs GMC/WFME recommendations."),("Shift length reviews","Cap at 12h, ensure 11h rest between clinical shifts."),("Wellbeing days","2-3/semester replacing teaching with optional stress management workshops.")]),
        ("📚","Academic dissatisfaction","Burnout, Depression",[("Curriculum feedback loops","Termly surveys with visible action reports."),("Early clinical exposure","Meaningful patient contact from Year 1 (Dornan et al., 2006)."),("Career counselling","Dedicated advisory service, especially for Years 3-4."),("Effort-grade transparency","Clear marking criteria, detailed feedback on assessments.")]),
        ("🌊","Stressful life events","Depression, SI",[("Proactive check-ins","Welfare conversations triggered by reported life events."),("Emergency counselling slots","20% of capacity reserved for same-week appointments."),("Flexible assessment","Mitigating circumstances without invasive documentation."),("Financial hardship funds","Rapid-access grants (€200-500) to prevent cascading stress.")]),
        ("🍷","Substance use","Depression, Anxiety",[("Confidential screening","Universal AUDIT-C in routine health checks."),("Harm reduction","Factual information, no moralistic messaging."),("Alternative social spaces","Attractive non-alcohol events and venues."),("Peer-led awareness","Student ambassadors trained in motivational interviewing.")]),
        ("⚡","State anxiety","All four targets",[("Exam anxiety workshops","CBT-based sessions before high-stakes periods."),("Assessment redesign","Spread across the year with formative feedback."),("MBSR programmes","8-week mindfulness courses as electives (Daya & Hearn, 2018)."),("Clinical simulation","Pre-rotation safe-failure sessions to reduce anticipatory anxiety.")]),
    ]
    for icon, factor, shap_t, items in PB:
        st.markdown(f'<div class="content-card"><div class="coping-header"><div class="badge">{icon}</div><div><h4 style="margin:0;">{factor}</h4><span style="font-size:11px;color:#6b7a8d;">Targets: {shap_t}</span></div></div>', unsafe_allow_html=True)
        for t, d in items:
            st.markdown(f'<div style="margin-left:50px;margin-bottom:10px;"><p style="font-size:14px;color:#0F2B3D;font-weight:600;margin-bottom:2px;">✦ {t}</p><p style="font-size:13px;color:#3a4a5e;line-height:1.6;margin:0;">{d}</p></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    nav_back_to_hub()


# ============================================================================
# FEATURE 3: RESOURCE FINDER
# ============================================================================

def page_resources():
    render_top_bar()
    role = st.session_state.get("role","student")
    hub = "Student hub" if role=="student" else "Institution hub"
    hub_key = "student_hub" if role=="student" else "institution_hub"
    render_breadcrumb(("MedMind","gateway"),(hub, hub_key),"Resource finder")

    st.markdown('<div style="padding:16px 0 8px;"><h3 style="font-size:20px;font-weight:600;color:#0F2B3D;">Resource finder</h3><p style="font-size:13px;color:#6b7a8d;">Mental health support across Spain — national and university-specific.</p></div>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">National resources — always available</p>', unsafe_allow_html=True)
    for c, icon, title, desc, action, href, clr in zip(st.columns(3),
        ["📞","🏥","💻"],["024 Crisis Line","PAIME","Telefono de la Esperanza"],
        ["Free, confidential, 24/7.","Specialised support for medical students.","Emotional support, 24/7. Call 717 003 717."],
        ["Call 024","Visit PAIME","Call 717 003 717"],
        ["tel:024","https://www.icomem.es/seccion/SALUD-MENTAL-MEDICO","tel:717003717"],
        ["#a32d2d","#0F2B3D","#48a8c9"]):
        with c: st.markdown(f'<div class="content-card" style="text-align:center;border-top:3px solid {clr};"><div style="font-size:28px;margin-bottom:8px;">{icon}</div><h4>{title}</h4><p>{desc}</p><a href="{href}" target="_blank" style="display:inline-block;background:{clr};color:#fff;padding:8px 24px;border-radius:8px;text-decoration:none;font-weight:500;margin-top:8px;">{action}</a></div>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">University-specific resources</p>', unsafe_allow_html=True)
    sel = st.selectbox("Select your university", list(UNIVERSITY_RESOURCES.keys()), key="uni_sel")
    info = UNIVERSITY_RESOURCES[sel]
    web = f"<a href='{info['w']}' target='_blank'>{info['w']}</a>" if info['w']!="—" else "—"
    st.markdown(f'<div class="content-card"><h4>🏫 {sel}</h4><p><strong>Counselling:</strong> {info["c"]}</p><p><strong>Phone:</strong> {info["p"]}</p><p><strong>Web:</strong> {web}</p><p><strong>Student organisations:</strong> {info["s"]}</p><p><strong>PAIME office:</strong> {info["paime"]}</p></div>', unsafe_allow_html=True)
    nav_back_to_hub()


# ============================================================================
# FEATURE 4: COPING STRATEGIES
# ============================================================================

def page_coping():
    render_top_bar()
    render_breadcrumb(("MedMind","gateway"),("Student hub","student_hub"),"Coping strategies")
    if "pages_visited" not in st.session_state: st.session_state.pages_visited = []
    if "coping" not in st.session_state.pages_visited: st.session_state.pages_visited.append("coping")
    st.markdown('<div style="padding:16px 0 8px;"><h3 style="font-size:20px;font-weight:600;color:#0F2B3D;">Coping strategy library</h3><p style="font-size:13px;color:#6b7a8d;">Evidence-based techniques tailored for medical training.</p></div>', unsafe_allow_html=True)

    cats = {
        "Exhaustion & burnout": ("🔥",[("Micro-recovery breaks","5-minute breaks every 90 minutes. Step away, stretch, look out a window (Ariga & Lleras, 2011)."),("Sleep hygiene","Consistent bedtime even during exams. No screens 30 min before. Sleep is the most powerful burnout recovery tool."),("Boundary setting","Clear 'off' hours — no studying after 9pm, or one full day off per week."),("Energy management","Track energy across the day for a week. Schedule demanding tasks at peak energy."),("Protected exercise","30 min moderate exercise 3x/week. Effect size comparable to mild antidepressants (Schuch et al., 2016).")]),
        "Anxiety & worry": ("⚡",[("4-7-8 breathing","Inhale 4s, hold 7s, exhale 8s. Repeat 3x. Activates parasympathetic nervous system."),("5-4-3-2-1 grounding","5 things you see, 4 touch, 3 hear, 2 smell, 1 taste. Anchors you in the present."),("Worry time","15 min/day designated for worrying. Postpone anxious thoughts outside this window."),("Study scheduling","Specific tasks per block, not vague 'study pathology.' Concrete plans reduce anxiety."),("Cognitive defusion","Reframe 'I'm going to fail' → 'I'm having the thought that I might fail.'")]),
        "Loneliness & isolation": ("🫂",[("Micro-steps","One message to one person today. One connection per week builds momentum."),("Study groups","Even 2 people. Shared academic struggle + social contact is uniquely powerful."),("Shared vulnerability","Mention to a trusted peer that things have been hard. They almost certainly feel the same."),("Structured activities","Join one extracurricular — sports, choir, volunteering. Automatic social contact.")]),
        "Career doubt": ("🧭",[("Values clarification","Write why you chose medicine. Has it changed, or been obscured by the grind?"),("Mentor conversations","20 minutes with a clinician you admire can be more powerful than formal advice."),("Specialty exploration","If this rotation feels wrong, medicine has enormous variety. Shadow other specialties."),("Expectation mapping","Write expected vs experienced. Makes dissatisfaction concrete and addressable.")]),
        "Grief & loss": ("🕊️",[("Permission to grieve","Grief doesn't pause for exams. Academic dips are normal and usually recover."),("Expressive writing","15 min/day for 4 days about the loss. Robust evidence base (Pennebaker)."),("Mitigating circumstances","File early. Don't wait for results. Most universities are understanding."),("Professional support","If grief persists 6+ months with significant impairment, targeted therapy helps.")]),
    }

    # Adaptive default based on screening results
    cat_list = list(cats.keys())
    default_idx = 0
    ctx = get_personalization_context()
    if ctx["has_screening"] and ctx["top_risk"]:
        risk_to_cat = {"burnout": "Exhaustion & burnout", "anxiety": "Anxiety & worry", "depression": "Loneliness & isolation"}
        target_cat = risk_to_cat.get(ctx["top_risk"])
        if target_cat in cat_list:
            default_idx = cat_list.index(target_cat)
            st.markdown(f"""<div style="background:#D6EDF2;border:1px solid #48A8C9;border-radius:12px;padding:14px 18px;margin-bottom:14px;display:flex;align-items:center;gap:10px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#48A8C9" stroke-width="1.8" style="flex-shrink:0;"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
                <span style="font-size:13px;color:#48A8C9;">We've pre-selected <strong>{target_cat}</strong> based on your screening. You can browse other topics below.</span>
            </div>""", unsafe_allow_html=True)

    sel = st.selectbox("Choose a topic", cat_list, index=default_idx, key="cop_cat")
    icon, strats = cats[sel]
    st.markdown(f'<div class="coping-header" style="margin-top:16px;"><div class="badge">{icon}</div><h4 style="font-size:18px;font-weight:600;color:#0F2B3D;margin:0;">{sel}</h4></div>', unsafe_allow_html=True)
    for t, d in strats:
        st.markdown(f'<div class="content-card"><h4>{t}</h4><p>{d}</p></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:12px;color:#8c96a4;text-align:center;margin-top:8px;">These complement — but don\'t replace — professional support. Call 024 if struggling.</div>', unsafe_allow_html=True)
    nav_back_to_hub()


# ============================================================================
# FEATURE 5: SELF-MONITORING TRACKER
# ============================================================================

TQ = [{"id":"t_exhaustion","text":"How exhausted have you felt?","scale":["Not at all","Slightly","Moderately","Very","Extremely"]},
      {"id":"t_anxiety","text":"How anxious have you felt?","scale":["Not at all","Slightly","Moderately","Very","Extremely"]},
      {"id":"t_mood","text":"Overall mood?","scale":["Very low","Low","Okay","Good","Very good"]},
      {"id":"t_social","text":"How connected to others?","scale":["Very isolated","Somewhat isolated","Okay","Connected","Very connected"]},
      {"id":"t_academic","text":"Academic satisfaction?","scale":["Not at all","Slightly","Moderately","Quite","Very"]},
      {"id":"t_sleep","text":"Sleep quality?","scale":["Very poor","Poor","Okay","Good","Very good"]},
      {"id":"t_motivation","text":"Motivation for medicine?","scale":["Not at all","Slightly","Moderately","Quite","Very"]},
      {"id":"t_coping","text":"How well are you coping?","scale":["Very poorly","Poorly","Okay","Well","Very well"]}]

def page_tracker():
    render_top_bar()
    render_breadcrumb(("MedMind","gateway"),("Student hub","student_hub"),"Monthly tracker")
    if "pages_visited" not in st.session_state: st.session_state.pages_visited = []
    if "tracker" not in st.session_state.pages_visited: st.session_state.pages_visited.append("tracker")
    st.markdown('<div style="padding:16px 0 8px;"><h3 style="font-size:20px;font-weight:600;color:#0F2B3D;">Self-monitoring tracker</h3><p style="font-size:13px;color:#6b7a8d;">Monthly check-in. Sign in with a username on the student hub to save your history across sessions.</p></div>', unsafe_allow_html=True)
    user = st.session_state.get("user_profile", "")
    if "tracker_history" not in st.session_state:
        st.session_state.tracker_history = load_user_data(user, "tracker") or [] if user else []

    # Calendar reminder
    rc1, rc2 = st.columns([3, 1])
    with rc1:
        st.markdown("""<div style="background:#EEEDFE;border:1px solid #CECBF6;border-radius:12px;padding:14px 18px;margin-bottom:12px;display:flex;align-items:center;gap:10px;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3C3489" stroke-width="1.8" style="flex-shrink:0;"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>
            <span style="font-size:13px;color:#3C3489;">Add a monthly reminder to your calendar so you don't forget check-ins.</span>
        </div>""", unsafe_allow_html=True)
    with rc2:
        st.download_button(
            label="Download reminder",
            data=generate_tracker_ics(),
            file_name="medmind-monthly-reminder.ics",
            mime="text/calendar",
            key="tracker_ics",
            use_container_width=True,
        )

    tab1, tab2 = st.tabs(["New check-in", "My trend"])
    with tab1:
        st.markdown('<div class="content-card"><h4>Monthly check-in</h4><p>Rate each based on the <strong>past 4 weeks</strong>.</p></div>', unsafe_allow_html=True)
        resp = {}
        for q in TQ: resp[q["id"]] = st.select_slider(q["text"], q["scale"], q["scale"][2], key=f"trk_{q['id']}")
        if st.button("Save check-in", key="save_ck", use_container_width=True):
            from datetime import datetime
            entry = {"date": datetime.now().strftime("%Y-%m-%d"), "responses": {q["id"]: q["scale"].index(resp[q["id"]]) for q in TQ}}
            inv = ["t_exhaustion","t_anxiety"]
            adj = {k: (4-v) if k in inv else v for k,v in entry["responses"].items()}
            entry["overall_score"] = round(np.mean(list(adj.values())), 2)
            st.session_state.tracker_history.append(entry)
            if user: save_user_data(user, "tracker", st.session_state.tracker_history)
            st.success("Saved." + (" Synced to your profile." if user else " Tip: sign in on the student hub to keep this across sessions."))
    with tab2:
        hist = st.session_state.tracker_history
        if not hist:
            st.markdown('<div class="content-card" style="text-align:center;"><p style="color:#6b7a8d;">No check-ins yet.</p></div>', unsafe_allow_html=True)
        else:
            import matplotlib.pyplot as plt; import matplotlib; matplotlib.use('Agg')
            dates=[h["date"] for h in hist]; scores=[h["overall_score"] for h in hist]
            fig,ax=plt.subplots(figsize=(10,4)); fig.patch.set_alpha(0); ax.set_facecolor('none')
            ax.axhspan(0,1.5,alpha=0.08,color='#e24b4a'); ax.axhspan(1.5,2.5,alpha=0.08,color='#ef9f27'); ax.axhspan(2.5,4,alpha=0.08,color='#5dcaa5')
            ax.plot(range(len(dates)),scores,'o-',color='#0F2B3D',linewidth=2.5,markersize=8,zorder=5)
            for i,(d,s) in enumerate(zip(dates,scores)):
                c='#e24b4a' if s<1.5 else '#ef9f27' if s<2.5 else '#5dcaa5'
                ax.plot(i,s,'o',color=c,markersize=10,zorder=6)
            ax.set_xticks(range(len(dates))); ax.set_xticklabels(dates,rotation=45,ha='right',fontsize=9)
            ax.set_ylabel('Wellbeing',fontsize=11,color='#5a6a7e'); ax.set_ylim(-0.2,4.2)
            ax.set_yticks([0,1,2,3,4]); ax.set_yticklabels(['Very low','Low','Moderate','Good','Very good'],fontsize=9)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); fig.tight_layout(); st.pyplot(fig); plt.close()
            if st.button("Clear all data", key="clr_trk"):
                st.session_state.tracker_history=[]
                if user: save_user_data(user, "tracker", [])
                st.rerun()
    nav_back_to_hub()


# ============================================================================
# FEATURE 6: EARLY WARNING GUIDE
# ============================================================================

def page_earlywarning():
    render_top_bar()
    render_breadcrumb(("MedMind","gateway"),("Institution hub","institution_hub"),"Early warning guide")
    st.markdown('<div style="padding:16px 0 8px;"><h3 style="font-size:20px;font-weight:600;color:#0F2B3D;">Early warning system — implementation guide</h3><p style="font-size:13px;color:#6b7a8d;">Step-by-step framework for systematic mental health screening.</p></div>', unsafe_allow_html=True)

    steps = [
        ("01","Frequency","Screen <strong>annually</strong> (start of year) + at <strong>transition points</strong> (Year 1 entry, clinical rotation start, pre-finals). MedMind's 35-item form takes under 5 minutes."),
        ("02","Questionnaire","35-item screening achieves AUCs of 0.891 (depression), 0.746 (burnout), 0.857 (anxiety), 0.828 (SI) — <em>without</em> clinical instruments. Full 42-feature set available for 15-20 min diagnostic sessions."),
        ("03","Referral protocols","<strong>Green:</strong> Wellbeing resources, re-screen in 6 months. <strong>Amber:</strong> Voluntary counselling referral, email within 48h, follow up at 2 weeks. <strong>Red:</strong> Same-day contact from welfare lead. SI flags trigger soft crisis protocol with 024 + PAIME."),
        ("04","Communications","Pre-screening: normalise ('as part of our commitment to wellbeing'). Emphasise confidential, voluntary, no academic impact. Results: immediate individual, annual aggregate. Follow-up: warm language ('check in with you', not 'flagged')."),
        ("05","Privacy & ethics","Data minimisation, anonymisation (suppress <30 cohorts), explicit consent, ethics committee review with student reps, GDPR-compliant with appointed DPO."),
        ("06","Integration","Tutorial system (aggregate trends), academic progress committees (pattern ID, never punitive), student health services (direct pathway), student unions (CEEM in governance), national reporting."),
    ]
    for n, t, c in steps:
        st.markdown(f'<div class="content-card" style="border-left:4px solid #0F2B3D;"><div style="display:flex;align-items:flex-start;gap:16px;"><div style="min-width:44px;height:44px;background:#0F2B3D;color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:14px;">{n}</div><div><h4 style="margin-top:8px;">{t}</h4><p>{c}</p></div></div></div>', unsafe_allow_html=True)
    nav_back_to_hub()


# ============================================================================
# FEATURE 7: COST CALCULATOR
# ============================================================================

def page_calculator():
    render_top_bar()
    render_breadcrumb(("MedMind","gateway"),("Institution hub","institution_hub"),"Cost calculator")
    st.markdown('<div style="padding:16px 0 8px;"><h3 style="font-size:20px;font-weight:600;color:#0F2B3D;">Cost-of-inaction calculator</h3><p style="font-size:13px;color:#6b7a8d;">Model untreated mental health costs vs screening investment.</p></div>', unsafe_allow_html=True)

    c1,c2=st.columns(2)
    with c1:
        n=st.number_input("Total medical students",50,5000,800,50,key="cn"); rp=st.slider("% at risk",20,70,40,key="cr"); dr=st.slider("Dropout/repeat rate for at-risk (%)",5,40,15,key="cd")
    with c2:
        cpy=st.number_input("Annual cost/student (EUR)",2000,30000,8000,500,key="cc"); ey=st.slider("Extra years per repeater",0.5,3.0,1.0,0.5,key="ce"); sc=st.number_input("Screening cost/student/year (EUR)",5,200,25,5,key="cs")

    nr=n*rp/100; nd=nr*dr/100; ce=nd*cpy*ey; cl=nd*cpy*0.5; ti=ce+cl; st_=n*sc
    sl=ti*0.30; sh=ti*0.50

    st.markdown('<p class="section-header">Results</p>', unsafe_allow_html=True)
    for col,val,lab,clr in zip(st.columns(3),[f"€{ti:,.0f}",f"€{st_:,.0f}",f"€{sl:,.0f}–€{sh:,.0f}"],["Cost of inaction","Screening cost","Estimated savings"],["#a32d2d","#1b6b8a","#0f6e56"]):
        with col: st.markdown(f'<div class="metric-card" style="border-top:3px solid {clr};"><div class="m-value" style="color:{clr};">{val}</div><div class="m-label">{lab}</div></div>', unsafe_allow_html=True)

    import matplotlib.pyplot as plt; import matplotlib; matplotlib.use('Agg')
    fig,ax=plt.subplots(figsize=(10,4)); fig.patch.set_alpha(0); ax.set_facecolor('none')
    cats_=['Inaction','Screening','Net savings\n(conservative)','Net savings\n(optimistic)']
    vals_=[ti,st_,sl-st_,sh-st_]; clrs=['#e24b4a','#48a8c9','#5dcaa5','#0f6e56']
    bars=ax.bar(cats_,vals_,color=clrs,edgecolor='white',width=0.55)
    for b,v in zip(bars,vals_): ax.text(b.get_x()+b.get_width()/2.,b.get_height()+max(vals_)*0.02,f'€{v:,.0f}',ha='center',fontsize=10,fontweight='500',color='#0F2B3D')
    ax.set_ylabel('EUR',fontsize=11,color='#5a6a7e');ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    fig.tight_layout();st.pyplot(fig);plt.close()

    st.markdown('<div class="content-card" style="border-left:4px solid #ef9f27;"><h4>⚠️ Limitations</h4><p>Simplified assumptions. 30-50% reduction based on Eisenberg et al. (2009), Hunt &amp; Eisenberg (2010). Real costs include presenteeism, faculty time, and human costs.</p></div>', unsafe_allow_html=True)
    nav_back_to_hub()


# ============================================================================
# FEATURE 8: PEER STORIES
# ============================================================================

def page_stories():
    render_top_bar()
    render_breadcrumb(("MedMind","gateway"),("Student hub","student_hub"),"Peer stories")
    if "pages_visited" not in st.session_state: st.session_state.pages_visited = []
    if "stories" not in st.session_state.pages_visited: st.session_state.pages_visited.append("stories")
    st.markdown('<div style="padding:16px 0 8px;"><h3 style="font-size:20px;font-weight:600;color:#0F2B3D;">Peer stories</h3><p style="font-size:13px;color:#6b7a8d;">Composite narratives from medical student experiences. Filter by what feels relevant to you.</p></div>', unsafe_allow_html=True)

    stories = [
        ("I was in Year 3 when I first realised something was wrong. I'd always managed everything — top of the class, social life, sports. But halfway through my first rotation, I started dreading every morning.", "Year 3, clinical onset — burnout, delayed help-seeking", "Accessed PAIME, 8 CBT sessions, returned with a self-care plan.", 3, "burnout", "clinical transition"),
        ("The transition to clinical rotations was the hardest period of my life. I felt invisible and incompetent. The residents were too busy to teach.", "Year 4, clinical transition — career doubt, alienation", "Mentor conversations rebuilt confidence. Discovered paediatrics.", 4, "depression", "clinical transition"),
        ("I didn't think I needed help until my parents said they barely recognised me. I'd been telling myself everyone in medicine feels this way.", "Year 5, accumulated burnout — self-denial", "PAIME support, treatment for moderate depression, continued with adjustments.", 5, "burnout", "accumulated stress"),
        ("I come from a small town where being a doctor is the ultimate achievement. When I couldn't cope, I felt ashamed.", "Year 2, first-generation — shame, isolation", "Peer mentorship was transformative. Later became a peer mentor.", 2, "depression", "isolation"),
        ("My father was diagnosed with cancer during second year. I tried to keep everything going.", "Year 2, bereavement — grief, resilience mythology", "Supported leave, returned the following year.", 2, "anxiety", "life events"),
        ("As a non-heterosexual student in a traditional medical school, I had to keep part of myself hidden.", "Year 3, LGBTQ+ identity — minority stress", "Helped establish an LGBTQ+ network.", 3, "anxiety", "identity"),
    ]

    # Filters
    c1, c2, c3 = st.columns(3)
    with c1: year_filter = st.selectbox("Year", ["All years", "Years 1-2", "Years 3-4", "Years 5-6"], key="story_year")
    with c2: risk_filter = st.selectbox("Risk area", ["All", "Burnout", "Depression", "Anxiety"], key="story_risk")
    with c3: theme_filter = st.selectbox("Theme", ["All", "Clinical transition", "Isolation", "Life events", "Identity", "Accumulated stress"], key="story_theme")

    filtered = stories
    if year_filter == "Years 1-2": filtered = [s for s in filtered if s[3] <= 2]
    elif year_filter == "Years 3-4": filtered = [s for s in filtered if 3 <= s[3] <= 4]
    elif year_filter == "Years 5-6": filtered = [s for s in filtered if s[3] >= 5]
    if risk_filter != "All": filtered = [s for s in filtered if s[4] == risk_filter.lower()]
    if theme_filter != "All": filtered = [s for s in filtered if s[5] == theme_filter.lower()]

    if not filtered:
        st.markdown('<div class="content-card" style="text-align:center;"><p style="color:#888780;">No stories match these filters. Try broadening your selection.</p></div>', unsafe_allow_html=True)
    for q, attr, outcome, yr, risk, theme in filtered:
        st.markdown(f'<div class="story-card"><div class="quote">"{q}"</div><div class="attribution">{attr}</div></div>', unsafe_allow_html=True)
        with st.expander("What happened next?"):
            st.markdown(f'<p style="font-size:14px;color:#3a4a5e;line-height:1.65;">{outcome}</p>', unsafe_allow_html=True)

    st.markdown('<div class="content-card" style="text-align:center;border-top:3px solid #48A8C9;margin-top:16px;"><h4>You are not alone</h4><p>Seeking support is a sign of strength.</p><p style="margin-top:12px;"><a href="tel:024" style="display:inline-block;background:#0F2B3D;color:#fff;padding:10px 28px;border-radius:10px;text-decoration:none;font-weight:500;">024</a>&nbsp;&nbsp;<a href="https://www.icomem.es/seccion/SALUD-MENTAL-MEDICO" target="_blank" style="display:inline-block;background:#48A8C9;color:#fff;padding:10px 28px;border-radius:10px;text-decoration:none;font-weight:500;">PAIME</a></p></div>', unsafe_allow_html=True)
    nav_back_to_hub()


# ============================================================================
# MINDGUIDE AGENT
# ============================================================================

AGENT_SYS = """You are MindGuide, a supportive AI in MedMind for Spanish medical students.
NEVER diagnose. NEVER mention suicidal ideation — if self-harm comes up, provide 024 + PAIME.
Be warm, concise (2-4 paragraphs), respond in the student's language. Like a wise older student who cares.
When the student has screening results, reference them naturally. If they've visited specific pages in the app, mention relevant tools they've already seen."""

def get_agent_context():
    a = st.session_state.get("answers", {}); r = st.session_state.get("screening_results", {})
    pv = st.session_state.get("pages_visited", [])
    p = [f"Year {a.get('course_year','?')}, {'female' if a.get('gender_female',0)==1 else 'male'}"]
    for t, v in r.items():
        if t != "suicidal_ideation": p.append(f"{t.replace('_',' ').title()}: {get_risk_level(v)[0]} ({v:.0%})")
    if pv:
        page_labels = {"results":"screening results","coping":"coping strategies","resources":"resource finder","tracker":"monthly tracker","mir_student":"MIR preparation hub","benchmark":"national benchmarks","sleep_tracker":"sleep tracker","letter":"letter to future self","exam_mode":"exam period mode","stories":"peer stories"}
        visited = [page_labels.get(pg, pg) for pg in pv]
        p.append(f"Pages visited: {', '.join(visited)}")
    return "\n".join(p)

def call_openai(msgs, sys):
    try:
        import openai; return openai.OpenAI().chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys},*msgs], max_tokens=500, temperature=0.7).choices[0].message.content
    except: return None

def fallback_response(msg):
    m = msg.lower()
    if any(w in m for w in ["suicide","kill","die","suicidio","matarme","self-harm","hurt myself"]): return "I hear you. Please reach out to **024** right now — free, confidential, 24/7. **PAIME** also offers specialised support. You don't have to go through this alone."
    if any(w in m for w in ["tired","exhausted","burnout","cansad"]): return "Exhaustion is incredibly common in medical training. Protecting one evening per week, setting study boundaries, or even a 15-minute walk can help. If it feels overwhelming, your counselling service can help build a structured plan."
    if any(w in m for w in ["anxious","anxiety","worried","stressed","ansiedad"]): return "About 1 in 4 medical students experience elevated anxiety. 4-7-8 breathing, smaller study blocks, and identifying specific triggers all help. A counsellor can develop targeted strategies if it's interfering with studies or sleep."
    if any(w in m for w in ["sad","depress","triste","hopeless"]): return "What you're feeling takes courage to acknowledge. PAIME (024) offers free, confidential help for medical students. You've already taken a step by reflecting — that matters."
    if any(w in m for w in ["alone","lonely","solo","aislad"]): return "Isolation during training is more common than most realise. One concrete step: a study group or peer support programme. Even one regular contact makes a difference."
    return "Thank you for sharing. Is there something specific — workload, support network, emotions — you'd like to explore? I'm here to help you think things through."

def page_agent():
    render_top_bar()
    render_breadcrumb(("MedMind","gateway"),("Student hub","student_hub"),("Results","results"),"MindGuide")
    st.markdown('<div style="text-align:center;padding:20px 0 12px;"><h3 style="font-size:20px;font-weight:600;color:#0F2B3D;">💬 MindGuide</h3><p style="font-size:13px;color:#6b7a8d;max-width:480px;margin:0 auto;">A supportive space to reflect. Not therapy — for ongoing support, speak with a professional.</p></div>', unsafe_allow_html=True)
    st.markdown('<div style="background:#f8f9fb;border:1px solid #E5DCC5;border-radius:10px;padding:10px 16px;margin-bottom:14px;text-align:center;"><p style="font-size:11px;color:#8c96a4;margin:0;">🔒 Completely private. Nothing stored. Disappears when you leave.</p></div>', unsafe_allow_html=True)

    if "chat_messages" not in st.session_state:
        st.session_state.agent_system = AGENT_SYS + f"\n\nPROFILE:\n{get_agent_context()}"
        st.session_state.chat_messages = [{"role":"assistant","content":"Hi — I've seen your results and I'm here if you'd like to talk. No pressure. Ask about strategies, resources, or just tell me what's on your mind."}]

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"], avatar="💬" if msg["role"]=="assistant" else "🧑‍⚕️"):
            st.markdown(msg["content"])
    inp = st.chat_input("Type your message...")
    if inp:
        st.session_state.chat_messages.append({"role":"user","content":inp})
        with st.chat_message("user", avatar="🧑‍⚕️"): st.markdown(inp)
        with st.chat_message("assistant", avatar="💬"):
            with st.spinner(""):
                r = call_openai([{"role":m["role"],"content":m["content"]} for m in st.session_state.chat_messages], st.session_state.agent_system) or fallback_response(inp)
                st.markdown(r); st.session_state.chat_messages.append({"role":"assistant","content":r})

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back to results", key="ag_res"):
            st.session_state.chat_messages = []; st.session_state.page = "results"; st.rerun()
    with c2:
        if st.button("← Back to student hub", key="ag_hub"):
            st.session_state.chat_messages = []; st.session_state.page = "student_hub"; st.session_state.answers = {}; st.rerun()


# ============================================================================
# NEW FEATURE: PERSONALISED ACTION PLAN (shown after results)
# ============================================================================

def page_action_plan():
    render_top_bar()
    render_breadcrumb(("MedMind","gateway"),("Student hub","student_hub"),("Results","results"),"Your action plan")
    results = st.session_state.get("screening_results", {})
    answers = st.session_state.get("answers", {})
    if not results:
        st.warning("Complete the screening first to generate your action plan.")
        nav_back_to_hub(); return

    vis = {k: v for k, v in results.items() if k != 'suicidal_ideation'}
    ht = max(vis, key=vis.get) if vis else 'depression'
    hp = vis.get(ht, 0)
    level = get_risk_level(hp)[0].lower()
    year = answers.get("course_year", 3)
    support = answers.get("social_support_n", 3)
    satisfaction = answers.get("acad_satisfaction_choice", 3)
    n_events = sum(answers.get(f"eve_{i}", 0) for i in range(1, 6))

    st.markdown(f'<div style="padding:16px 0 8px;"><h3 style="font-size:20px;font-weight:600;color:#0F2B3D;">Your personalised action plan</h3><p style="font-size:13px;color:#6b7a8d;">Based on your screening profile — tailored to your year, risk factors, and circumstances.</p></div>', unsafe_allow_html=True)

    # Generate contextual steps
    steps = []

    # Step 1: always — based on highest risk
    if level == "high":
        steps.append(("🔴", "This week", "Book a counselling appointment", f"Your {ht.replace('_',' ')} indicators are elevated. The single most impactful step is speaking with a professional. Most university services offer same-week appointments. If your university's service has a waiting list, contact PAIME directly at 024 — they can see you faster."))
    elif level == "moderate":
        steps.append(("🟡", "This week", "Take one concrete step", f"Your profile shows moderate {ht.replace('_',' ')} indicators. Choose one: tell a trusted person how you're feeling, visit the coping strategies library in this app, or look up your university counselling service in our resource finder."))
    else:
        steps.append(("🟢", "This week", "Protect what's working", "Your profile looks relatively stable. Focus on maintaining the habits that are keeping you well — social connections, sleep, and whatever boundaries you've set around study time."))

    # Step 2: based on social support
    if support <= 2:
        steps.append(("👥", "This month", "Strengthen your support network", "Your social support is in the lower range. One concrete action: reach out to one person this week — a classmate, an old friend, a family member. If you don't know where to start, look for a study group or peer mentorship programme through your faculty."))
    elif satisfaction <= 2:
        steps.append(("📚", "This month", "Reconnect with your motivation", f"As a Year {year} student feeling unsatisfied with medicine, consider booking a career counselling session. Also try the values clarification exercise in our coping library — write down why you chose medicine and what's changed."))
    else:
        steps.append(("📈", "This month", "Set up self-monitoring", "Start tracking your wellbeing monthly using the tracker in this app. Even a 2-minute check-in once a month helps you spot patterns before they become problems."))

    # Step 3: based on year and events
    if year >= 4:
        steps.append(("🏥", "Ongoing", "Navigate the clinical transition", f"Year {year} is clinically intensive. Prioritise: buddy up with a rotation partner, set a hard boundary on shift recovery time, and use our exam period mode during high-stress weeks."))
    elif n_events >= 2:
        steps.append(("🌊", "Ongoing", "Get support for life stressors", f"You're managing {n_events} recent life events alongside medical school. This is a lot. Consider filing for mitigating circumstances proactively, and keep your personal tutor informed — they can't help if they don't know."))
    else:
        steps.append(("🧘", "Ongoing", "Build a sustainable routine", "You're in a relatively stable period. This is the best time to build habits that protect you during harder stretches: regular exercise, consistent sleep, and at least one non-medical activity per week."))

    for emoji, timing, title, desc in steps:
        st.markdown(f"""<div class="content-card" style="border-left:4px solid #48a8c9;">
            <div style="display:flex;align-items:flex-start;gap:16px;">
                <div style="font-size:28px;">{emoji}</div>
                <div><span style="font-size:11px;font-weight:600;color:#48a8c9;text-transform:uppercase;letter-spacing:0.5px;">{timing}</span>
                <h4 style="margin:4px 0 6px;">{title}</h4><p>{desc}</p></div>
            </div></div>""", unsafe_allow_html=True)

    # Quick links
    st.markdown('<p class="section-header">Quick links from your plan</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📍 Find my resources", key="ap_res", use_container_width=True):
            st.session_state.page = "resources"; st.rerun()
    with c2:
        if st.button("💡 Coping strategies", key="ap_cop", use_container_width=True):
            st.session_state.page = "coping"; st.rerun()
    with c3:
        if st.button("📈 Start tracking", key="ap_trk", use_container_width=True):
            st.session_state.page = "tracker"; st.rerun()

    nav_back_to_hub()


# ============================================================================
# NEW FEATURE: "AM I NORMAL?" BENCHMARK
# ============================================================================

def page_benchmark():
    render_top_bar()
    render_breadcrumb(("MedMind","gateway"),("Student hub","student_hub"),"Am I normal?")
    if "pages_visited" not in st.session_state: st.session_state.pages_visited = []
    if "benchmark" not in st.session_state.pages_visited: st.session_state.pages_visited.append("benchmark")
    df = load_dashboard_data()
    answers = st.session_state.get("answers", {})

    st.markdown('<div style="padding:16px 0 8px;"><h3 style="font-size:20px;font-weight:600;color:#0F2B3D;">Am I normal?</h3><p style="font-size:13px;color:#6b7a8d;">See where your responses fall compared to 5,216 medical students nationwide. Complete the screening first, or explore the population data directly.</p></div>', unsafe_allow_html=True)

    if df is None:
        st.warning("Benchmark data not available (dabe_clean_full.csv required).")
        nav_back_to_hub(); return

    if not answers:
        st.info("Complete the screening first to see your personal benchmarks. In the meantime, here are the population distributions.")

    comparisons = [
        ("social_support_n", "Social support network size", "Number of close support people", {1: "None", 2: "1-2", 3: "3-5", 4: "5+"}),
        ("acad_satisfaction_studies", "Study satisfaction", "Satisfaction with studies", {1: "Not at all", 2: "Somewhat", 3: "Quite", 4: "Very"}),
        ("acad_satisfaction_choice", "Career choice satisfaction", "Satisfaction with choosing medicine", {1: "Not at all", 2: "Somewhat", 3: "Quite", 4: "Very"}),
        ("attendance", "Class attendance", "Non-compulsory class attendance", None),
        ("alcohol", "Alcohol consumption", "Drinking frequency", {0: "Never", 1: "Occasionally", 2: "1-2x/week", 3: "2-3x/week", 4: "3+/week"}),
    ]

    import matplotlib.pyplot as plt; import matplotlib; matplotlib.use('Agg')

    for col_id, title, xlabel, labels in comparisons:
        if col_id not in df.columns:
            continue
        your_val = answers.get(col_id)
        year = answers.get("course_year")

        fig, ax = plt.subplots(figsize=(10, 3)); fig.patch.set_alpha(0); ax.set_facecolor('none')
        data = df[col_id].dropna()
        vals, counts = np.unique(data, return_counts=True)
        pcts = counts / counts.sum() * 100
        bar_colors = ['#d0dae8'] * len(vals)

        if your_val is not None:
            for i, v in enumerate(vals):
                if v == your_val:
                    bar_colors[i] = '#48a8c9'
            percentile = (data < your_val).mean() * 100

        bars = ax.bar(vals, pcts, color=bar_colors, edgecolor='white', width=0.6)
        for b, p in zip(bars, pcts):
            ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.5, f'{p:.0f}%', ha='center', fontsize=9, color='#5a6a7e')

        if labels:
            ax.set_xticks(list(labels.keys()))
            ax.set_xticklabels(list(labels.values()), fontsize=9)
        ax.set_ylabel('%', fontsize=10, color='#5a6a7e')
        ax.set_title(title, fontsize=13, color='#0F2B3D', fontweight='600', loc='left')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.tick_params(colors='#5a6a7e')

        if your_val is not None:
            ax.axvline(x=your_val, color='#0F2B3D', linestyle='--', linewidth=1.5, alpha=0.6)
            ax.text(your_val + 0.1, ax.get_ylim()[1]*0.85, f'You ({percentile:.0f}th percentile)', fontsize=10, color='#0F2B3D', fontweight='600')

        fig.tight_layout(); st.pyplot(fig); plt.close()

    if answers:
        # N problems comparison
        your_probs = answers.get("n_problems", 0)
        avg_probs = df['n_problems'].mean() if 'n_problems' in df.columns else 0
        st.markdown(f"""<div class="content-card" style="text-align:center;">
            <h4>Number of perceived difficulties</h4>
            <div style="display:flex;justify-content:center;gap:40px;margin-top:12px;">
                <div><div style="font-size:36px;font-weight:600;color:#48a8c9;">{your_probs}</div><div style="font-size:12px;color:#6b7a8d;">YOUR SCORE</div></div>
                <div><div style="font-size:36px;font-weight:600;color:#0F2B3D;">{avg_probs:.1f}</div><div style="font-size:12px;color:#6b7a8d;">NATIONAL AVERAGE</div></div>
            </div></div>""", unsafe_allow_html=True)

    st.markdown('<div style="font-size:12px;color:#8c96a4;text-align:center;margin-top:12px;">All comparisons based on DABE 2020 (n=5,216). Individual variation is normal. These benchmarks provide context, not judgement.</div>', unsafe_allow_html=True)
    nav_back_to_hub()


# ============================================================================
# NEW FEATURE: EXAM PERIOD MODE
# ============================================================================

def page_exam_mode():
    render_top_bar()
    render_breadcrumb(("MedMind","gateway"),("Student hub","student_hub"),"Exam period mode")
    if "pages_visited" not in st.session_state: st.session_state.pages_visited = []
    if "exam_mode" not in st.session_state.pages_visited: st.session_state.pages_visited.append("exam_mode")

    st.markdown("""<div style="text-align:center;padding:24px 0 16px;">
        <div style="font-size:36px;margin-bottom:8px;">⚡</div>
        <h2 style="font-size:22px;font-weight:600;color:#0F2B3D;margin-bottom:6px;">Exam period mode</h2>
        <p style="font-size:14px;color:#6b7a8d;">Quick tools for when you're under pressure. Everything you need in 2 minutes.</p>
    </div>""", unsafe_allow_html=True)

    # Quick check-in
    st.markdown('<p class="section-header">30-second check-in</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: stress = st.select_slider("Stress level right now", ["1 — Calm","2","3 — Moderate","4","5 — Overwhelmed"], "3 — Moderate", key="ex_stress")
    with c2: sleep = st.select_slider("Hours slept last night", ["<4","4-5","5-6","6-7","7-8","8+"], "6-7", key="ex_sleep")
    with c3: eaten = st.select_slider("Have you eaten today?", ["Nothing","Snack","One meal","Two+ meals"], "One meal", key="ex_eaten")

    stress_val = ["1 — Calm","2","3 — Moderate","4","5 — Overwhelmed"].index(stress)
    sleep_val = ["<4","4-5","5-6","6-7","7-8","8+"].index(sleep)

    # Dynamic guidance
    if stress_val >= 3:
        st.markdown('<div class="content-card" style="border-left:4px solid #e24b4a;"><h4>🔴 High stress detected</h4><p>You\'re running hot. Before you do anything else: <strong>4-7-8 breathing, right now.</strong> Inhale 4 seconds, hold 7, exhale 8. Do it 3 times. Then come back.</p></div>', unsafe_allow_html=True)
    if sleep_val <= 1:
        st.markdown('<div class="content-card" style="border-left:4px solid #ef9f27;"><h4>🟡 Sleep deficit</h4><p>Under 5 hours impairs memory consolidation — the thing you need most during exams. Tonight, set a non-negotiable bedtime. 6 hours of sleep + focused study beats 3 hours of sleep + exhausted cramming.</p></div>', unsafe_allow_html=True)
    if eaten == "Nothing":
        st.markdown('<div class="content-card" style="border-left:4px solid #ef9f27;"><h4>🟡 Eat something</h4><p>Your brain uses 20% of your calories. Even a banana and some nuts will help concentration. Step away from the desk for 5 minutes to eat.</p></div>', unsafe_allow_html=True)

    # Quick coping tools
    st.markdown('<p class="section-header">Quick tools</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="content-card">
            <h4>🫁 4-7-8 Breathing</h4>
            <p>1. Inhale through nose: <strong>4 seconds</strong><br>
            2. Hold breath: <strong>7 seconds</strong><br>
            3. Exhale through mouth: <strong>8 seconds</strong><br>
            Repeat 3 times. Takes 57 seconds total.</p></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="content-card">
            <h4>🎯 Next-hour plan</h4>
            <p>Write down exactly ONE topic to study in the next hour. Not "revise cardiology" — something like "learn the 5 causes of heart failure." Specific = less anxiety.</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="content-card">
            <h4>🧊 5-4-3-2-1 Grounding</h4>
            <p>Name: <strong>5</strong> things you see · <strong>4</strong> you can touch ·
            <strong>3</strong> you hear · <strong>2</strong> you smell · <strong>1</strong> you taste.<br>
            Anchors you in the present. 30 seconds.</p></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="content-card">
            <h4>⏱️ Pomodoro reset</h4>
            <p>25 min focused study → 5 min break (stand up, stretch, water).
            After 4 cycles → 20 min real break. This works better than marathon sessions.</p></div>""", unsafe_allow_html=True)

    # Crisis
    st.markdown("""<div class="content-card" style="text-align:center;border-top:3px solid #a32d2d;margin-top:8px;">
        <h4>If you're in crisis right now</h4>
        <p>Call <strong>024</strong> — free, confidential, 24/7. You matter more than any exam.</p>
        <a href="tel:024" style="display:inline-block;background:#a32d2d;color:#fff;padding:10px 28px;border-radius:8px;text-decoration:none;font-weight:500;margin-top:8px;">📞 Call 024 now</a>
    </div>""", unsafe_allow_html=True)
    nav_back_to_hub()


# ============================================================================
# NEW FEATURE: SLEEP & ROUTINE TRACKER
# ============================================================================

def page_sleep_tracker():
    render_top_bar()
    render_breadcrumb(("MedMind","gateway"),("Student hub","student_hub"),"Sleep tracker")
    if "pages_visited" not in st.session_state: st.session_state.pages_visited = []
    if "sleep_tracker" not in st.session_state.pages_visited: st.session_state.pages_visited.append("sleep_tracker")
    st.markdown('<div style="padding:16px 0 8px;"><h3 style="font-size:20px;font-weight:600;color:#0F2B3D;">Sleep & routine tracker</h3><p style="font-size:13px;color:#6b7a8d;">3 inputs per day. Sign in on the student hub to save across sessions.</p></div>', unsafe_allow_html=True)

    user = st.session_state.get("user_profile", "")
    if "sleep_log" not in st.session_state:
        st.session_state.sleep_log = load_user_data(user, "sleep") or [] if user else []

    tab1, tab2 = st.tabs(["📝 Log today", "📈 My week"])
    with tab1:
        st.markdown('<div class="content-card"><h4>Today\'s entry</h4></div>', unsafe_allow_html=True)
        hours = st.slider("Hours slept last night", 0.0, 12.0, 7.0, 0.5, key="sl_hours")
        mood = st.select_slider("How's your mood right now?", ["1 — Very low","2 — Low","3 — Okay","4 — Good","5 — Great"], "3 — Okay", key="sl_mood")
        kindness = st.text_input("One thing you did (or will do) for yourself today:", placeholder="e.g. took a 20-minute walk, called a friend, went to bed on time...", key="sl_kind")

        if st.button("Save entry", key="sl_save", use_container_width=True):
            from datetime import datetime
            mood_val = ["1 — Very low","2 — Low","3 — Okay","4 — Good","5 — Great"].index(mood) + 1
            st.session_state.sleep_log.append({"date": datetime.now().strftime("%a %d/%m"), "hours": hours, "mood": mood_val, "kindness": kindness or "—"})
            if user: save_user_data(user, "sleep", st.session_state.sleep_log)
            st.success("Saved." + (" Synced to your profile." if user else ""))

    with tab2:
        log = st.session_state.sleep_log
        if not log:
            st.markdown('<div class="content-card" style="text-align:center;"><p style="color:#6b7a8d;">No entries yet. Log today to start seeing patterns.</p></div>', unsafe_allow_html=True)
        else:
            import matplotlib.pyplot as plt; import matplotlib; matplotlib.use('Agg')
            dates = [e["date"] for e in log]
            hours_ = [e["hours"] for e in log]
            moods = [e["mood"] for e in log]

            fig, ax1 = plt.subplots(figsize=(10, 4)); fig.patch.set_alpha(0); ax1.set_facecolor('none')
            x = range(len(dates))
            # Sleep bars
            bar_colors = ['#e24b4a' if h < 5 else '#ef9f27' if h < 6.5 else '#48a8c9' for h in hours_]
            ax1.bar(x, hours_, color=bar_colors, alpha=0.7, width=0.5, label='Sleep (hours)')
            ax1.set_ylabel('Hours slept', fontsize=11, color='#48a8c9')
            ax1.set_ylim(0, 12)
            ax1.axhline(y=7, color='#48a8c9', linestyle=':', alpha=0.4)

            # Mood line on secondary axis
            ax2 = ax1.twinx()
            ax2.plot(x, moods, 'o-', color='#534ab7', linewidth=2, markersize=8, label='Mood')
            ax2.set_ylabel('Mood', fontsize=11, color='#534ab7')
            ax2.set_ylim(0.5, 5.5)
            ax2.set_yticks([1,2,3,4,5])
            ax2.set_yticklabels(['Very low','Low','Okay','Good','Great'], fontsize=8)

            ax1.set_xticks(list(x)); ax1.set_xticklabels(dates, fontsize=9, rotation=45, ha='right')
            ax1.spines['top'].set_visible(False); ax2.spines['top'].set_visible(False)
            fig.legend(loc='upper right', fontsize=9, bbox_to_anchor=(0.95, 0.95))
            fig.tight_layout(); st.pyplot(fig); plt.close()

            # Self-care log
            st.markdown('<p class="section-header">Self-care log</p>', unsafe_allow_html=True)
            for e in reversed(log[-7:]):
                clr = '#e24b4a' if e["hours"]<5 else '#ef9f27' if e["hours"]<6.5 else '#5dcaa5'
                st.markdown(f'<div style="display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid #eef1f5;"><span style="font-size:12px;color:#6b7a8d;min-width:70px;">{e["date"]}</span><span style="font-size:12px;font-weight:600;color:{clr};min-width:50px;">{e["hours"]}h</span><span style="font-size:12px;color:#534ab7;min-width:30px;">☺ {e["mood"]}/5</span><span style="font-size:13px;color:#3a4a5e;">{e["kindness"]}</span></div>', unsafe_allow_html=True)

            if st.button("Clear log", key="sl_clear"):
                st.session_state.sleep_log = []
                if user: save_user_data(user, "sleep", [])
                st.rerun()

    nav_back_to_hub()


# ============================================================================
# NEW FEATURE: LETTER TO YOUR FUTURE SELF
# ============================================================================

def page_letter():
    render_top_bar()
    render_breadcrumb(("MedMind","gateway"),("Student hub","student_hub"),"Letter to yourself")
    if "pages_visited" not in st.session_state: st.session_state.pages_visited = []
    if "letter" not in st.session_state.pages_visited: st.session_state.pages_visited.append("letter")
    st.markdown("""<div style="text-align:center;padding:24px 0 16px;">
        <div style="font-size:36px;margin-bottom:8px;">✉️</div>
        <h2 style="font-size:22px;font-weight:600;color:#0F2B3D;margin-bottom:6px;">Letter to your future self</h2>
        <p style="font-size:14px;color:#6b7a8d;max-width:520px;margin:0 auto;">Write a message during a good moment. Read it when things feel hard.<br>
        Research on self-compassion shows this simple exercise builds resilience.</p>
    </div>""", unsafe_allow_html=True)

    user = st.session_state.get("user_profile", "")
    if "letter" not in st.session_state:
        st.session_state.letter = load_user_data(user, "letter") if user else None

    tab1, tab2 = st.tabs(["Write", "Read"])
    with tab1:
        st.markdown("""<div class="content-card">
            <h4>Prompts to get you started</h4>
            <p>• What would you tell yourself on a day when everything feels impossible?<br>
            • What are you proud of right now that you might forget later?<br>
            • What do you know about yourself that you need to be reminded of?<br>
            • What would your best friend say to you right now?</p></div>""", unsafe_allow_html=True)

        letter_text = st.text_area("Your letter:", height=200, placeholder="Dear future me,\n\nI'm writing this on a good day because I know there will be hard ones...", key="letter_input")
        if st.button("Save letter", key="save_letter", use_container_width=True):
            if letter_text.strip():
                from datetime import datetime
                st.session_state.letter = {"text": letter_text, "date": datetime.now().strftime("%B %d, %Y")}
                if user: save_user_data(user, "letter", st.session_state.letter)
                st.success("Saved." + (" Synced to your profile." if user else " Sign in on the student hub to keep it permanently."))
            else:
                st.warning("Write something first — even a single sentence counts.")

    with tab2:
        if st.session_state.letter:
            l = st.session_state.letter
            st.markdown(f"""<div class="story-card" style="border-left-color:#534ab7;">
                <div style="font-size:12px;color:#6b7a8d;margin-bottom:12px;">Written on {l['date']}</div>
                <div class="quote" style="font-style:normal;white-space:pre-wrap;">{l['text']}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown('<div style="text-align:center;padding:12px 0;"><p style="font-size:13px;color:#6b7a8d;font-style:italic;">You wrote this during a good moment. Trust that version of yourself.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="content-card" style="text-align:center;"><p style="color:#6b7a8d;">No letter yet. Write one during a good moment — your future self will thank you.</p></div>', unsafe_allow_html=True)

    nav_back_to_hub()


# ============================================================================
# NEW FEATURE: INSTITUTIONAL BENCHMARK COMPARISON
# ============================================================================

def page_inst_benchmark():
    render_top_bar()
    render_breadcrumb(("MedMind","gateway"),("Institution hub","institution_hub"),"Benchmark comparison")
    st.markdown('<div style="padding:16px 0 8px;"><h3 style="font-size:20px;font-weight:600;color:#0F2B3D;">Benchmark your university</h3><p style="font-size:13px;color:#6b7a8d;">Enter your institution\'s data (actual or estimated) and compare against DABE 2020 national averages.</p></div>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">Your university data</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: dep = st.number_input("Depression (%)", 0.0, 100.0, 25.0, 1.0, key="bm_dep")
    with c2: burn = st.number_input("Burnout (%)", 0.0, 100.0, 40.0, 1.0, key="bm_burn")
    with c3: anx = st.number_input("Anxiety (%)", 0.0, 100.0, 25.0, 1.0, key="bm_anx")
    with c4: si = st.number_input("Suicidal ideation (%)", 0.0, 100.0, 12.0, 1.0, key="bm_si")

    national = {"Depression": 23.4, "Burnout": 36.8, "Anxiety": 22.8, "Suicidal ideation": 10.6}
    yours = {"Depression": dep, "Burnout": burn, "Anxiety": anx, "Suicidal ideation": si}

    st.markdown('<p class="section-header">Comparison</p>', unsafe_allow_html=True)

    import matplotlib.pyplot as plt; import matplotlib; matplotlib.use('Agg')
    fig, ax = plt.subplots(figsize=(10, 5)); fig.patch.set_alpha(0); ax.set_facecolor('none')
    conds = list(national.keys())
    x = np.arange(len(conds)); w = 0.3
    nat_vals = [national[c] for c in conds]; your_vals = [yours[c] for c in conds]
    b1 = ax.bar(x - w/2, nat_vals, w, label='DABE 2020 national', color='#48a8c9', edgecolor='white')
    b2 = ax.bar(x + w/2, your_vals, w, label='Your university', color='#0F2B3D', edgecolor='white')
    for b in b1: ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.8, f'{b.get_height():.1f}%', ha='center', fontsize=10, color='#48a8c9')
    for b in b2: ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.8, f'{b.get_height():.1f}%', ha='center', fontsize=10, color='#0F2B3D', fontweight='600')
    ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=12); ax.set_ylabel('Prevalence (%)', fontsize=11, color='#5a6a7e')
    ax.legend(fontsize=10); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    # Insights
    for cond in conds:
        diff = yours[cond] - national[cond]
        if diff > 5:
            st.markdown(f'<div class="content-card" style="border-left:4px solid #e24b4a;"><h4>⚠️ {cond}: {diff:+.1f} pp above national average</h4><p>Your {cond.lower()} rate ({yours[cond]:.1f}%) is meaningfully above the national figure ({national[cond]:.1f}%). Consider targeted intervention — see the Intervention Playbook for evidence-based actions.</p></div>', unsafe_allow_html=True)
        elif diff < -5:
            st.markdown(f'<div class="content-card" style="border-left:4px solid #5dcaa5;"><h4>✓ {cond}: {diff:+.1f} pp below national average</h4><p>Your {cond.lower()} rate is lower than the national benchmark. Identify what\'s working and protect those factors.</p></div>', unsafe_allow_html=True)

    nav_back_to_hub()


# ============================================================================
# NEW FEATURE: PROGRAMME EVALUATION FRAMEWORK
# ============================================================================

def page_evaluation():
    render_top_bar()
    render_breadcrumb(("MedMind","gateway"),("Institution hub","institution_hub"),"Programme evaluation")
    st.markdown('<div style="padding:16px 0 8px;"><h3 style="font-size:20px;font-weight:600;color:#0F2B3D;">Programme evaluation framework</h3><p style="font-size:13px;color:#6b7a8d;">How to measure whether your mental health interventions are working.</p></div>', unsafe_allow_html=True)

    # KPIs
    st.markdown('<p class="section-header">Key performance indicators</p>', unsafe_allow_html=True)
    kpis = [
        ("📊", "Screening participation rate", "Target: >60% of cohort", "Measures programme reach and student trust. Below 40% suggests communication or stigma barriers."),
        ("📉", "Prevalence change (year-on-year)", "Target: reduction or stabilisation", "Compare annual screening results. Expect 6-12 months before measurable change."),
        ("🔗", "Referral conversion rate", "Target: >50% of amber/red students engage", "Of students offered support, how many take it? Low conversion = accessibility or trust problem."),
        ("🔁", "Repeat screening rate", "Target: >40% re-screen at next cycle", "Students who re-screen trust the system. Declining rates signal disengagement."),
        ("📈", "At-risk trajectory improvement", "Target: >30% improve by next screen", "Among students flagged at-risk, what % show improvement at follow-up?"),
        ("🎓", "Academic outcome correlation", "Monitor: dropout/repeat rates in flagged group", "Compare academic outcomes for screened vs non-screened cohorts."),
        ("😌", "Student satisfaction with support", "Target: >4.0/5.0", "Post-intervention survey. Measures perceived quality of the support pathway."),
        ("⏱️", "Time to first appointment", "Target: <10 working days", "From flag to first professional contact. Longer waits correlate with lower engagement."),
    ]
    for icon, title, target, desc in kpis:
        st.markdown(f'<div class="content-card"><div style="display:flex;gap:14px;"><div style="font-size:24px;">{icon}</div><div><h4 style="margin:0 0 4px;">{title}</h4><span style="font-size:12px;font-weight:600;color:#48a8c9;">{target}</span><p style="margin-top:6px;">{desc}</p></div></div></div>', unsafe_allow_html=True)

    # Timeline
    st.markdown('<p class="section-header">Implementation timeline</p>', unsafe_allow_html=True)
    timeline = [
        ("Months 1-2", "Setup", "Ethics approval, questionnaire finalisation, communication strategy, staff training, IT integration."),
        ("Month 3", "Pilot", "Run with one cohort (e.g., Year 1). Gather feedback, refine workflow, test referral pathways."),
        ("Months 4-6", "Full launch", "Universal screening across all years. Track participation and early KPIs."),
        ("Month 9", "First review", "Analyse initial data: participation rates, referral volumes, student satisfaction. Adjust."),
        ("Month 12", "Annual report", "Full year-on-year comparison. Present to leadership with benchmark data and ROI estimate."),
        ("Year 2+", "Sustained monitoring", "Annual screening cycle established. Focus shifts to trajectory tracking and intervention refinement."),
    ]
    for period, phase, desc in timeline:
        st.markdown(f'<div style="display:flex;gap:16px;margin-bottom:12px;"><div style="min-width:100px;font-size:12px;font-weight:600;color:#48a8c9;">{period}</div><div><span style="font-size:14px;font-weight:600;color:#0F2B3D;">{phase}</span><p style="font-size:13px;color:#3a4a5e;margin:2px 0 0;">{desc}</p></div></div>', unsafe_allow_html=True)

    nav_back_to_hub()


# ============================================================================
# NEW FEATURE: RISK FACTOR HEATMAP
# ============================================================================

def page_heatmap():
    render_top_bar()
    render_breadcrumb(("MedMind","gateway"),("Institution hub","institution_hub"),"Risk factor heatmap")
    st.markdown('<div style="padding:16px 0 8px;"><h3 style="font-size:20px;font-weight:600;color:#0F2B3D;">Risk factor heatmap</h3><p style="font-size:13px;color:#6b7a8d;">Which risk factors peak at which point in training, broken down by demographics.</p></div>', unsafe_allow_html=True)

    df = load_dashboard_data()
    if df is None:
        st.warning("Dataset not available."); nav_back_to_hub(); return

    import matplotlib.pyplot as plt; import matplotlib; matplotlib.use('Agg')

    # Build heatmap data: factors x course years
    factors_to_plot = {
        'acad_satisfaction_studies': ('Study dissatisfaction (inverted)', True),
        'social_support_n': ('Low social support (inverted)', True),
        'n_problems': ('Number of problems', False),
        'alcohol': ('Alcohol consumption', False),
        'takes_psychopharm': ('Psychiatric medication use', False),
    }

    # Add n_problems if missing
    if 'n_problems' not in df.columns:
        prob_cols = [c for c in df.columns if c.startswith('prob_')]
        if prob_cols: df['n_problems'] = df[prob_cols].sum(axis=1)

    gender_filter = st.selectbox("Filter by gender", ["All students", "Female only", "Male only"], key="hm_gender")
    if gender_filter == "Female only": df_f = df[df['gender'] == 'Female']
    elif gender_filter == "Male only": df_f = df[df['gender'] == 'Male']
    else: df_f = df

    available = {k: v for k, v in factors_to_plot.items() if k in df_f.columns}
    if not available:
        st.warning("Required columns not found in dataset."); nav_back_to_hub(); return

    # Build matrix
    years = sorted(df_f['course_year'].unique())
    labels = [v[0] for v in available.values()]
    matrix = np.zeros((len(available), len(years)))

    for i, (col, (_, invert)) in enumerate(available.items()):
        for j, y in enumerate(years):
            vals = df_f[df_f['course_year'] == y][col].dropna()
            if len(vals) > 0:
                mean_val = vals.mean()
                overall_mean = df_f[col].dropna().mean()
                overall_std = df_f[col].dropna().std()
                if overall_std > 0:
                    z = (mean_val - overall_mean) / overall_std
                    matrix[i, j] = -z if invert else z
                else:
                    matrix[i, j] = 0

    fig, ax = plt.subplots(figsize=(10, max(4, len(available) * 0.8 + 1)))
    fig.patch.set_alpha(0); ax.set_facecolor('none')
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=-1.5, vmax=1.5)
    ax.set_xticks(range(len(years))); ax.set_xticklabels([f'Year {y}' for y in years], fontsize=11)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=11)

    for i in range(len(labels)):
        for j in range(len(years)):
            v = matrix[i, j]
            color = 'white' if abs(v) > 0.8 else '#0F2B3D'
            ax.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=10, color=color, fontweight='500')

    plt.colorbar(im, ax=ax, label='Z-score (higher = more risk)', shrink=0.8)
    ax.set_title(f'Risk factors by course year ({gender_filter.lower()})', fontsize=14, color='#0F2B3D', fontweight='600', pad=16)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="content-card"><h4>Reading the heatmap</h4><p><strong>Red cells</strong> indicate above-average risk for that factor in that year. <strong>Green cells</strong> indicate below-average. Values are z-scores relative to the full cohort mean. For inverted scales (satisfaction, support), higher values = more risk. This helps identify precisely where and when to target interventions.</p></div>', unsafe_allow_html=True)
    nav_back_to_hub()


# ============================================================================
# NEW FEATURE: POLICY BRIEF GENERATOR
# ============================================================================

def page_policy_brief():
    render_top_bar()
    render_breadcrumb(("MedMind","gateway"),("Institution hub","institution_hub"),"Policy brief")
    st.markdown('<div style="padding:16px 0 8px;"><h3 style="font-size:20px;font-weight:600;color:#0F2B3D;">Policy brief generator</h3><p style="font-size:13px;color:#6b7a8d;">Generate a downloadable one-page summary for your dean or board.</p></div>', unsafe_allow_html=True)

    uni_name = st.text_input("University name:", "Universidad de ...", key="pb_uni")
    n_students = st.number_input("Number of medical students:", 100, 5000, 800, key="pb_n")

    if st.button("Generate policy brief", key="gen_pb", use_container_width=True):
        brief_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
body {{ font-family: 'Helvetica Neue', Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 40px; color: #0F2B3D; font-size: 14px; line-height: 1.6; }}
h1 {{ font-size: 22px; border-bottom: 3px solid #48a8c9; padding-bottom: 8px; }}
h2 {{ font-size: 16px; color: #48a8c9; margin-top: 24px; }}
.stat {{ display: inline-block; text-align: center; padding: 12px 20px; margin: 6px; background: #f4f7fb; border-radius: 8px; }}
.stat .num {{ font-size: 28px; font-weight: 700; color: #0F2B3D; }}
.stat .lbl {{ font-size: 11px; color: #6b7a8d; text-transform: uppercase; }}
.rec {{ background: #eef3f9; border-left: 3px solid #48a8c9; padding: 10px 16px; margin: 8px 0; border-radius: 0 6px 6px 0; }}
.footer {{ margin-top: 32px; padding-top: 12px; border-top: 1px solid #E5DCC5; font-size: 11px; color: #8c96a4; }}
</style></head><body>
<h1>Mental Health Screening: The Case for Action at {uni_name}</h1>
<p><em>Policy brief prepared by MedMind — {pd.Timestamp.now().strftime('%B %Y')}</em></p>

<h2>The Problem</h2>
<p>National data from 5,216 Spanish medical students (DABE 2020) reveals:</p>
<div class="stat"><div class="num">23.4%</div><div class="lbl">Depression</div></div>
<div class="stat"><div class="num">36.8%</div><div class="lbl">Burnout</div></div>
<div class="stat"><div class="num">22.8%</div><div class="lbl">Anxiety</div></div>
<div class="stat"><div class="num">10.6%</div><div class="lbl">Suicidal ideation</div></div>
<p>Post-COVID estimates are substantially higher (depression 48%, anxiety 45% — Lin & Saragih 2024). At {uni_name}, with approximately {n_students} medical students, this means an estimated <strong>{int(n_students*0.37)}</strong> students currently experiencing burnout and <strong>{int(n_students*0.23)}</strong> with clinically significant depression.</p>

<h2>The Cost of Inaction</h2>
<p>Assuming 40% of students are at risk and 15% of those drop out or repeat a year at EUR 8,000/year:</p>
<div class="stat"><div class="num">EUR {int(n_students*0.4*0.15*8000*1.5):,}</div><div class="lbl">Estimated annual cost</div></div>
<p>This does not include presenteeism, faculty time, reputational impact, or human suffering.</p>

<h2>The Solution: Systematic Screening</h2>
<p>MedMind provides a validated, 5-minute screening tool that identifies at-risk students without requiring clinical instruments. Test AUCs: depression 0.891, anxiety 0.857, burnout 0.746.</p>

<h2>Recommendations</h2>
<div class="rec"><strong>1.</strong> Implement annual universal screening at the start of each academic year.</div>
<div class="rec"><strong>2.</strong> Add transition-point screening at clinical rotation entry (Year 3-4).</div>
<div class="rec"><strong>3.</strong> Establish a tiered referral protocol (green/amber/red) with a named welfare lead.</div>
<div class="rec"><strong>4.</strong> Ring-fence counselling capacity: reserve 20% for urgent same-week appointments.</div>
<div class="rec"><strong>5.</strong> Publish annual transparency reports to build student trust.</div>

<h2>Investment Required</h2>
<div class="stat"><div class="num">EUR {int(n_students*25):,}</div><div class="lbl">Annual screening cost (est. EUR 25/student)</div></div>
<div class="stat"><div class="num">150-250%</div><div class="lbl">Expected ROI</div></div>

<div class="footer">
<p>Generated by MedMind — Capstone project by Maria Reiter Hernandez, IE University, 2026. Supervised by Luis Angel Galindo.<br>
Data source: DABE 2020 (n=5,216, 43 universities). Literature: Lin & Saragih 2024; Hawsawi et al. 2025; Soler et al. 2025.<br>
Crisis support: 024 (free, 24/7) · PAIME: <a href="https://www.icomem.es/seccion/SALUD-MENTAL-MEDICO">icomem.es</a></p>
</div></body></html>"""

        st.download_button("📄 Download policy brief (HTML)", brief_html, file_name=f"MedMind_Policy_Brief_{uni_name.replace(' ','_')}.html", mime="text/html", use_container_width=True)
        st.markdown('<div class="content-card" style="border-left:4px solid #5dcaa5;"><h4>✓ Brief generated</h4><p>Click the download button above. The HTML file can be opened in any browser and printed as a PDF. It\'s formatted for single-page A4 printing.</p></div>', unsafe_allow_html=True)

    nav_back_to_hub()


# ============================================================================
# NEW FEATURE: SIMULATED REPORTING DASHBOARD
# ============================================================================

def page_sim_dashboard():
    render_top_bar()
    render_breadcrumb(("MedMind","gateway"),("Institution hub","institution_hub"),"Simulated dashboard")
    st.markdown('<div style="padding:16px 0 8px;"><h3 style="font-size:20px;font-weight:600;color:#0F2B3D;">Simulated reporting dashboard</h3><p style="font-size:13px;color:#6b7a8d;">What institutional screening data looks like in practice. This uses realistic synthetic data — no real students.</p></div>', unsafe_allow_html=True)

    import matplotlib.pyplot as plt; import matplotlib; matplotlib.use('Agg')
    np.random.seed(42)

    n_total = 820; participation = 0.72; n_screened = int(n_total * participation)
    months = ['Sep','Oct','Nov','Dec','Jan','Feb']
    monthly_screens = [380, 120, 85, 65, 95, 75]

    # Metric cards
    st.markdown('<p class="section-header">Semester overview (simulated)</p>', unsafe_allow_html=True)
    for c, v, l in zip(st.columns(5), [str(n_total), f"{participation*100:.0f}%", str(n_screened), "18.2%", "62%"],
                       ["Enrolled","Participation","Screened","High risk","Referral uptake"]):
        with c: st.markdown(f'<div class="metric-card"><div class="m-value">{v}</div><div class="m-label">{l}</div></div>', unsafe_allow_html=True)

    # Participation over time
    st.markdown('<p class="section-header">Monthly screening volume</p>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 3.5)); fig.patch.set_alpha(0); ax.set_facecolor('none')
    ax.bar(months, monthly_screens, color='#48a8c9', edgecolor='white')
    for i, v in enumerate(monthly_screens): ax.text(i, v+5, str(v), ha='center', fontsize=11, color='#0F2B3D', fontweight='500')
    ax.set_ylabel('Screenings', fontsize=11, color='#5a6a7e')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    # Risk tier breakdown
    st.markdown('<p class="section-header">Risk tier distribution</p>', unsafe_allow_html=True)
    tiers = {'Low': int(n_screened*0.52), 'Moderate': int(n_screened*0.30), 'High': int(n_screened*0.18)}
    tier_colors = {'Low': '#5dcaa5', 'Moderate': '#ef9f27', 'High': '#e24b4a'}
    fig, ax = plt.subplots(figsize=(10, 3)); fig.patch.set_alpha(0); ax.set_facecolor('none')
    bars = ax.barh(list(tiers.keys()), list(tiers.values()), color=[tier_colors[k] for k in tiers], height=0.5)
    for b, v in zip(bars, tiers.values()):
        ax.text(b.get_width()+5, b.get_y()+b.get_height()/2., f'{v} students ({v/n_screened*100:.0f}%)', va='center', fontsize=11)
    ax.set_xlabel('Number of students', fontsize=11, color='#5a6a7e')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    # Referral funnel
    st.markdown('<p class="section-header">Referral funnel</p>', unsafe_allow_html=True)
    funnel = [("Flagged (moderate + high)", tiers['Moderate']+tiers['High']),
              ("Offered referral", int((tiers['Moderate']+tiers['High'])*0.95)),
              ("Accepted referral", int((tiers['Moderate']+tiers['High'])*0.62)),
              ("Attended first session", int((tiers['Moderate']+tiers['High'])*0.48)),
              ("Completed 3+ sessions", int((tiers['Moderate']+tiers['High'])*0.31))]
    max_v = funnel[0][1]
    for label, val in funnel:
        pct = val/max_v*100
        st.markdown(f'<div style="margin-bottom:8px;"><div style="display:flex;justify-content:space-between;"><span style="font-size:13px;color:#3a4a5e;">{label}</span><span style="font-size:13px;font-weight:600;color:#0F2B3D;">{val} ({pct:.0f}%)</span></div><div style="height:16px;background:#e8ecf1;border-radius:4px;overflow:hidden;"><div style="height:100%;width:{pct:.0f}%;background:linear-gradient(90deg,#0F2B3D,#48a8c9);border-radius:4px;"></div></div></div>', unsafe_allow_html=True)

    # Year breakdown
    st.markdown('<p class="section-header">High-risk students by year</p>', unsafe_allow_html=True)
    year_risk = {'Year 1': 15.2, 'Year 2': 14.8, 'Year 3': 22.1, 'Year 4': 24.5, 'Year 5': 18.3, 'Year 6': 13.6}
    fig, ax = plt.subplots(figsize=(10, 3.5)); fig.patch.set_alpha(0); ax.set_facecolor('none')
    bars = ax.bar(year_risk.keys(), year_risk.values(), color=['#48a8c9' if v < 20 else '#ef9f27' if v < 23 else '#e24b4a' for v in year_risk.values()], edgecolor='white')
    for b in bars: ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.5, f'{b.get_height():.1f}%', ha='center', fontsize=10, color='#0F2B3D')
    ax.set_ylabel('% flagged high-risk', fontsize=11, color='#5a6a7e'); ax.set_ylim(0, 30)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="content-card" style="border-left:4px solid #534ab7;"><h4>🔮 About this simulation</h4><p>All data on this page is synthetically generated to demonstrate what an institutional reporting dashboard would look like after one semester of screening. The numbers are calibrated to be realistic based on DABE 2020 prevalence rates and published intervention uptake figures. No real student data is shown.</p></div>', unsafe_allow_html=True)
    nav_back_to_hub()


# ============================================================================
# MIR: STUDENT PAGE
# ============================================================================

def page_mir_student():
    render_top_bar()
    render_breadcrumb(("MedMind","gateway"),("Student hub","student_hub"),"MIR preparation")
    if "pages_visited" not in st.session_state: st.session_state.pages_visited = []
    if "mir_student" not in st.session_state.pages_visited: st.session_state.pages_visited.append("mir_student")

    st.markdown("""<div style="padding:28px 0 16px;">
        <h2 style="font-family:'Fraunces',serif;font-size:24px;font-weight:600;color:#0F2B3D;margin-bottom:8px;">MIR preparation and wellbeing</h2>
        <p style="font-size:14px;color:#5a6a7e;line-height:1.65;max-width:600px;">
        The MIR isn't just another exam. It determines your specialty, your city, and the next chapter of your life.
        Your mental health during this period matters as much as your study score — and the two are more connected than you think.</p>
    </div>""", unsafe_allow_html=True)

    tabs = st.tabs(["Your wellness roadmap", "Specialty alignment", "Resources and tools", "Study planner", "MIR stories", "After the MIR"])

    # ═══ TAB 1: WELLNESS ROADMAP ═══
    with tabs[0]:
        st.markdown("""<div class="content-card" style="border-left:3px solid #534ab7;margin-top:12px;">
            <p style="font-family:'Fraunces',serif;font-size:15px;color:#0F2B3D;font-style:italic;line-height:1.7;margin:0;">
            "I wish someone had told me that the students who rest strategically outperform the ones who study the most hours.
            My best simulacro scores came after the weeks where I slept properly and saw friends."
            <span style="font-style:normal;color:#6b7a8d;"> — Year 6 student, matched into cardiology</span></p>
        </div>""", unsafe_allow_html=True)

        # Timeline cards — compact, warm, not bullet-heavy
        phases = [
            ("#e1f5ee", "#085041", "8-6 months out", "Build your foundation",
             "Set sustainable hours (8-10 focused beats 14 exhausted). Schedule one non-negotiable weekly social commitment. Choose your environment deliberately — academia structure vs home flexibility. Start tracking sleep now; this is where memory consolidation happens."),
            ("#fdf4e7", "#854f0b", "5-3 months out", "Watch for the wall",
             "This is where burnout peaks. If your motivation drops or you start dreading the desk, that's a signal — not weakness. Take one full rest day per week (your scores will actually improve). Stop comparing your hours or simulacro results with others. Check in with your body: eating, moving, seeing daylight?"),
            ("#fcebeb", "#a32d2d", "Final 2 months", "Protect your performance",
             "Slightly reduce study hours in the last 2 weeks — exhaustion-cramming does more harm than good. The night before the exam, your only job is sleep. Anxiety spikes are physiologically normal before high-stakes events; they don't mean you're unprepared. Have a logistics plan ready: transport, documents, food, water."),
            ("#eee9f5", "#3C3489", "After the exam", "Let yourself land",
             "The crash after months of adrenaline is real. Resist obsessive score-calculation. Reconnect with everything you postponed: people, rest, hobbies. If you feel persistently flat or empty — that's common and treatable, not a character flaw. PAIME and 024 are still available to you."),
        ]

        for bg, accent, timing, title, text in phases:
            st.markdown(f"""<div style="display:flex;gap:16px;margin:16px 0;">
                <div style="min-width:4px;background:{accent};border-radius:2px;"></div>
                <div style="flex:1;">
                    <span style="font-size:11px;font-weight:600;color:{accent};text-transform:uppercase;letter-spacing:0.5px;">{timing}</span>
                    <h4 style="font-family:'Fraunces',serif;font-size:16px;font-weight:500;color:#0F2B3D;margin:4px 0 6px;">{title}</h4>
                    <p style="font-size:13px;color:#3a4a5e;line-height:1.7;margin:0;">{text}</p>
                </div>
            </div>""", unsafe_allow_html=True)

        # One key insight card
        st.markdown("""<div class="content-card" style="margin-top:16px;text-align:center;">
            <p style="font-family:'Fraunces',serif;font-size:16px;color:#0F2B3D;line-height:1.7;margin:0;">
            The MIR tests breadth, not perfection.<br>You don't need to know everything. You need to know enough — and you almost certainly do.</p>
        </div>""", unsafe_allow_html=True)

    # ═══ TAB 2: SPECIALTY ALIGNMENT ═══
    with tabs[1]:
        st.markdown("""<div class="content-card" style="margin-top:12px;">
            <h4>Beyond the ranking number</h4>
            <p>The MIR ranking tells you what's available. It doesn't tell you what will make you happy.
            Students who choose based on genuine fit report lower burnout five years into residency than those
            who optimise for prestige. This tool helps you think about specialty through a wellbeing lens.</p>
        </div>""", unsafe_allow_html=True)

        q1 = st.select_slider("Patient relationships", ["I prefer brief, acute encounters", "I like a mix", "I want long-term relationships with patients"], "I like a mix", key="mir_q1")
        q2 = st.select_slider("Work rhythm", ["Fast-paced, high-adrenaline", "Variable — some intense, some calm", "Predictable, structured routine"], "Variable — some intense, some calm", key="mir_q2")
        q3 = st.select_slider("Procedures vs thinking", ["I want to work with my hands", "Balance of both", "I prefer clinical reasoning and diagnosis"], "Balance of both", key="mir_q3")
        q4 = st.select_slider("Schedule predictability", ["I don't mind unpredictable hours", "Some flexibility is fine", "I need a predictable schedule"], "Some flexibility is fine", key="mir_q4")
        q5 = st.select_slider("Team vs independence", ["I want to work in a close team", "Collaborative but with autonomy", "I prefer independent work"], "Collaborative but with autonomy", key="mir_q5")
        q6 = st.select_slider("What energises you most?", ["Solving a puzzle under pressure", "Building a relationship over time", "Mastering a technical skill"], "Building a relationship over time", key="mir_q6")

        if st.button("See my alignment", key="mir_align", use_container_width=True):
            q1_val = ["I prefer brief, acute encounters", "I like a mix", "I want long-term relationships with patients"].index(q1)
            q2_val = ["Fast-paced, high-adrenaline", "Variable — some intense, some calm", "Predictable, structured routine"].index(q2)
            q3_val = ["I want to work with my hands", "Balance of both", "I prefer clinical reasoning and diagnosis"].index(q3)
            q4_val = ["I don't mind unpredictable hours", "Some flexibility is fine", "I need a predictable schedule"].index(q4)
            q5_val = ["I want to work in a close team", "Collaborative but with autonomy", "I prefer independent work"].index(q5)
            q6_val = ["Solving a puzzle under pressure", "Building a relationship over time", "Mastering a technical skill"].index(q6)

            specialties = {
                "Emergency / ICU": {"desc": "Fast-paced acute care. Brief encounters, high adrenaline, team-based. Thriving under pressure and variety over continuity.", "w": {"q1":[3,1,0],"q2":[3,1,0],"q3":[1,2,1],"q4":[3,1,0],"q5":[3,1,0],"q6":[3,0,1]}},
                "Surgery": {"desc": "Procedural, technical mastery. Long training, demanding hours, tangible outcomes. Working with your hands and seeing immediate results.", "w": {"q1":[2,1,0],"q2":[2,2,0],"q3":[3,1,0],"q4":[2,1,0],"q5":[2,2,0],"q6":[1,0,3]}},
                "Internal medicine / Cardiology": {"desc": "Diagnostic reasoning, complex patients, longitudinal hospital care. Puzzle-solving and thinking through differential diagnoses.", "w": {"q1":[0,2,2],"q2":[1,3,0],"q3":[0,1,3],"q4":[1,2,1],"q5":[1,3,0],"q6":[3,1,0]}},
                "Family medicine / Primary care": {"desc": "Long-term relationships, whole-person care, predictable schedule. Continuity, breadth, and work-life balance.", "w": {"q1":[0,1,3],"q2":[0,1,3],"q3":[0,1,3],"q4":[0,1,3],"q5":[0,2,2],"q6":[0,3,0]}},
                "Paediatrics": {"desc": "Relationship-based, family-oriented, variable acuity. Connecting with families and balancing acute and chronic care.", "w": {"q1":[0,2,2],"q2":[1,3,0],"q3":[0,2,2],"q4":[1,2,1],"q5":[2,2,0],"q6":[0,3,0]}},
                "Psychiatry": {"desc": "Deep patient relationships, cognitive and relational work. Reflective personalities comfortable with ambiguity and emotional intensity.", "w": {"q1":[0,0,3],"q2":[0,1,3],"q3":[0,0,3],"q4":[0,1,3],"q5":[0,2,2],"q6":[0,3,0]}},
                "Radiology / Pathology": {"desc": "Independent, analytical, predictable hours. Structured environments, pattern recognition, less direct patient contact.", "w": {"q1":[2,1,0],"q2":[0,1,3],"q3":[0,1,3],"q4":[0,0,3],"q5":[0,1,3],"q6":[2,0,1]}},
                "Dermatology / Ophthalmology": {"desc": "Outpatient-focused, mix of procedures and clinic. Good lifestyle balance. Highly competitive MIR numbers.", "w": {"q1":[1,2,1],"q2":[0,1,3],"q3":[1,3,0],"q4":[0,1,3],"q5":[0,2,2],"q6":[0,1,3]}},
            }

            answers_vec = [q1_val, q2_val, q3_val, q4_val, q5_val, q6_val]
            q_keys = ["q1","q2","q3","q4","q5","q6"]
            scored = []
            for name, data in specialties.items():
                total = sum(data["w"][qk][answers_vec[i]] for i, qk in enumerate(q_keys))
                scored.append((name, data["desc"], total, total/18*100))
            scored.sort(key=lambda x: x[2], reverse=True)

            st.markdown('<p class="section-header">Your specialty alignment</p>', unsafe_allow_html=True)
            for rank, (spec, desc, score, pct) in enumerate(scored, 1):
                bar_color = "#0F2B3D" if rank <= 3 else "#c8d0dc"
                border = "border-left:3px solid #0F2B3D;" if rank <= 3 else ""
                rl = f'<span style="font-size:11px;font-weight:600;color:#fff;background:#0F2B3D;padding:2px 10px;border-radius:10px;">#{rank}</span>' if rank <= 3 else f'<span style="font-size:11px;color:#8c96a4;">#{rank}</span>'
                st.markdown(f"""<div style="padding:14px 18px;margin-bottom:8px;background:#fff;border:1px solid #E5DCC5;border-radius:8px;{border}">
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
                        <div style="display:flex;align-items:center;gap:10px;">{rl}<span style="font-family:'Fraunces',serif;font-size:15px;font-weight:500;color:#0F2B3D;">{spec}</span></div>
                        <span style="font-size:12px;font-weight:500;color:#5a6a7e;">{pct:.0f}% match</span>
                    </div>
                    <div style="height:6px;background:#eef1f5;border-radius:3px;overflow:hidden;margin-bottom:8px;"><div style="height:100%;width:{pct:.0f}%;background:{bar_color};border-radius:3px;"></div></div>
                    <p style="font-size:12px;color:#5a6a7e;margin:0;line-height:1.55;">{desc}</p>
                </div>""", unsafe_allow_html=True)

    # ═══ TAB 3: RESOURCES AND TOOLS ═══
    with tabs[2]:
        st.markdown('<p class="section-header">MIR preparation resources</p>', unsafe_allow_html=True)
        st.markdown("""<div class="content-card" style="margin-top:12px;">
            <p style="font-size:13px;color:#6b7a8d;line-height:1.6;">These are the tools and platforms most commonly used by Spanish MIR candidates.
            MedMind is not affiliated with any of them — this is a curated reference list based on student feedback.</p>
        </div>""", unsafe_allow_html=True)

        # Academies
        st.markdown("""<div style="margin-top:16px;">
            <h4 style="font-family:'Fraunces',serif;font-size:16px;color:#0F2B3D;margin-bottom:12px;">Preparation academies</h4>
        </div>""", unsafe_allow_html=True)

        academies = [
            ("MIR Asturias", "https://www.curso-mir.com/", "One of the largest. Comprehensive programme with simulacros, video classes, and presential/online options. Known for structured methodology."),
            ("AMIR", "https://www.academiamir.com/", "Widely used manuals and question banks. Strong online platform with adaptive learning. Offers both full courses and manual-only access."),
            ("CTO Medicina", "https://www.ctomedicina.com/", "Extensive question bank and video library. Hybrid format with in-person sessions in major cities."),
            ("Promir", "https://www.promir.es/", "Fully online platform. Question-based learning approach with detailed analytics on your performance patterns."),
            ("Grupo CTO", "https://www.grupocto.es/", "One of the oldest academies. Physical locations across Spain. Traditional classroom format with printed materials."),
        ]
        for name, url, desc in academies:
            st.markdown(f"""<div style="padding:12px 16px;margin-bottom:8px;background:#fff;border:1px solid #E5DCC5;border-radius:8px;display:flex;justify-content:space-between;align-items:flex-start;gap:16px;">
                <div>
                    <a href="{url}" target="_blank" style="font-size:14px;font-weight:500;color:#0F2B3D;text-decoration:none;border-bottom:1px solid #c8dbe8;">{name}</a>
                    <p style="font-size:12px;color:#5a6a7e;margin:4px 0 0;line-height:1.5;">{desc}</p>
                </div>
                <a href="{url}" target="_blank" style="font-size:11px;color:#3d8baa;text-decoration:none;white-space:nowrap;">Visit site</a>
            </div>""", unsafe_allow_html=True)

        # Past exams and study tools
        st.markdown("""<div style="margin-top:24px;">
            <h4 style="font-family:'Fraunces',serif;font-size:16px;color:#0F2B3D;margin-bottom:12px;">Past exams and question banks</h4>
        </div>""", unsafe_allow_html=True)

        tools = [
            ("Ministerio de Sanidad — Past MIR exams", "https://www.sanidad.gob.es/areas/formacionEspecializada/pruebasSelectivas/home.htm", "Official source. All past exams with answer keys, published by the Ministry."),
            ("MIR 2.0 (app)", "https://www.mir2.com/", "Popular mobile app for practising MIR questions on the go. Tracks progress by subject."),
            ("Pasaporte MIR", "https://www.pasaportemir.com/", "Free question bank with community explanations and discussion forums."),
            ("Desgloses MIR", "—", "Compiled past exam questions organised by specialty and topic. Available through most academies."),
        ]
        for name, url, desc in tools:
            link = f'<a href="{url}" target="_blank" style="font-size:11px;color:#3d8baa;text-decoration:none;">Visit</a>' if url != "—" else '<span style="font-size:11px;color:#8c96a4;">Included with academy</span>'
            name_html = f'<a href="{url}" target="_blank" style="font-size:14px;font-weight:500;color:#0F2B3D;text-decoration:none;border-bottom:1px solid #c8dbe8;">{name}</a>' if url != "—" else f'<span style="font-size:14px;font-weight:500;color:#0F2B3D;">{name}</span>'
            st.markdown(f"""<div style="padding:12px 16px;margin-bottom:8px;background:#fff;border:1px solid #E5DCC5;border-radius:8px;display:flex;justify-content:space-between;align-items:flex-start;gap:16px;">
                <div>{name_html}<p style="font-size:12px;color:#5a6a7e;margin:4px 0 0;line-height:1.5;">{desc}</p></div>{link}
            </div>""", unsafe_allow_html=True)

        # Community and support
        st.markdown("""<div style="margin-top:24px;">
            <h4 style="font-family:'Fraunces',serif;font-size:16px;color:#0F2B3D;margin-bottom:12px;">Community and support</h4>
        </div>""", unsafe_allow_html=True)

        community = [
            ("r/MIR (Reddit)", "https://www.reddit.com/r/mir/", "Spanish-language community. Advice on scheduling, academy comparisons, and emotional support from current candidates."),
            ("MIR student Telegram groups", "—", "Most academies have Telegram channels. Also look for your year's cohort group — search 'MIR [year]' on Telegram."),
            ("CEEM — Student support", "https://ceem.org.es/", "National medical student council. Resources on MIR preparation, advocacy, and student rights during the process."),
            ("PAIME", "https://www.icomem.es/seccion/SALUD-MENTAL-MEDICO", "If preparation is affecting your mental health, PAIME offers free, confidential support specifically for medical students. 024."),
        ]
        for name, url, desc in community:
            link = f'<a href="{url}" target="_blank" style="font-size:11px;color:#3d8baa;text-decoration:none;">Visit</a>' if url != "—" else '<span style="font-size:11px;color:#8c96a4;">Search on Telegram</span>'
            name_html = f'<a href="{url}" target="_blank" style="font-size:14px;font-weight:500;color:#0F2B3D;text-decoration:none;border-bottom:1px solid #c8dbe8;">{name}</a>' if url != "—" else f'<span style="font-size:14px;font-weight:500;color:#0F2B3D;">{name}</span>'
            st.markdown(f"""<div style="padding:12px 16px;margin-bottom:8px;background:#fff;border:1px solid #E5DCC5;border-radius:8px;display:flex;justify-content:space-between;align-items:flex-start;gap:16px;">
                <div>{name_html}<p style="font-size:12px;color:#5a6a7e;margin:4px 0 0;line-height:1.5;">{desc}</p></div>{link}
            </div>""", unsafe_allow_html=True)

    # ═══ TAB 4: STUDY PLANNER ═══
    with tabs[3]:
        st.markdown('<p class="section-header">MIR study planner</p>', unsafe_allow_html=True)
        st.markdown("""<div class="content-card" style="margin-top:12px;">
            <p>Log your weekly schedule, track what you've covered, and keep notes on what needs revision.
            Everything stays in this session — nothing is sent anywhere.</p>
        </div>""", unsafe_allow_html=True)

        mir_user = st.session_state.get("user_profile", "")
        if "mir_plan" not in st.session_state:
            st.session_state.mir_plan = load_user_data(mir_user, "mir_plan") or {"weeks": [], "subjects": {}} if mir_user else {"weeks": [], "subjects": {}}

        plan_tab1, plan_tab2 = st.tabs(["Weekly log", "Subject tracker"])

        with plan_tab1:
            st.markdown("""<div style="margin-top:12px;">
                <h4 style="font-family:'Fraunces',serif;font-size:15px;color:#0F2B3D;margin-bottom:12px;">Log this week</h4>
            </div>""", unsafe_allow_html=True)

            week_label = st.text_input("Week label (e.g. 'Week 12 — Jan 20-26')", key="mir_week_label")
            c1, c2 = st.columns(2)
            with c1:
                hours_studied = st.number_input("Total study hours this week", 0, 100, 40, key="mir_hours")
                simulacro_score = st.number_input("Simulacro score (if taken, 0 if none)", 0, 300, 0, key="mir_sim")
            with c2:
                subjects_covered = st.text_input("Subjects covered (comma-separated)", placeholder="e.g. Cardiology, Pneumology, Nephrology", key="mir_subjects")
                energy_level = st.select_slider("Energy/motivation this week", ["Very low", "Low", "Okay", "Good", "Great"], "Okay", key="mir_energy")

            notes = st.text_area("Notes to yourself", placeholder="What went well? What needs more work? How are you feeling?", height=80, key="mir_notes")

            if st.button("Save this week", key="mir_save_week", use_container_width=True):
                if week_label:
                    entry = {
                        "week": week_label,
                        "hours": hours_studied,
                        "simulacro": simulacro_score if simulacro_score > 0 else None,
                        "subjects": [s.strip() for s in subjects_covered.split(",") if s.strip()],
                        "energy": ["Very low","Low","Okay","Good","Great"].index(energy_level),
                        "notes": notes
                    }
                    st.session_state.mir_plan["weeks"].append(entry)
                    for s in entry["subjects"]:
                        if s not in st.session_state.mir_plan["subjects"]:
                            st.session_state.mir_plan["subjects"][s] = 0
                        st.session_state.mir_plan["subjects"][s] += 1
                    if mir_user: save_user_data(mir_user, "mir_plan", st.session_state.mir_plan)
                    st.success(f"Saved: {week_label}" + (" (synced)" if mir_user else " (sign in on hub to persist)"))
                else:
                    st.warning("Add a week label first.")

            # Show history
            weeks = st.session_state.mir_plan["weeks"]
            if weeks:
                st.markdown("""<div style="margin-top:20px;">
                    <h4 style="font-family:'Fraunces',serif;font-size:15px;color:#0F2B3D;margin-bottom:12px;">Your preparation log</h4>
                </div>""", unsafe_allow_html=True)

                # Trends
                if len(weeks) >= 2:
                    import matplotlib.pyplot as plt; import matplotlib; matplotlib.use('Agg')
                    fig, ax1 = plt.subplots(figsize=(10, 3.5)); fig.patch.set_alpha(0); ax1.set_facecolor('none')
                    wlabels = [w["week"][:12] for w in weeks]
                    hours = [w["hours"] for w in weeks]
                    energy = [w["energy"] for w in weeks]

                    ax1.bar(range(len(wlabels)), hours, color='#c8dbe8', width=0.5, label='Hours')
                    ax1.set_ylabel('Hours', fontsize=10, color='#5a6a7e')
                    ax1.set_ylim(0, max(hours)*1.3)

                    ax2 = ax1.twinx()
                    ax2.plot(range(len(wlabels)), energy, 'o-', color='#534ab7', linewidth=2, markersize=7, label='Energy')
                    ax2.set_ylabel('Energy', fontsize=10, color='#534ab7')
                    ax2.set_ylim(-0.5, 4.5)
                    ax2.set_yticks([0,1,2,3,4]); ax2.set_yticklabels(['V.low','Low','Okay','Good','Great'], fontsize=8)

                    ax1.set_xticks(range(len(wlabels))); ax1.set_xticklabels(wlabels, fontsize=8, rotation=45, ha='right')
                    ax1.spines['top'].set_visible(False); ax2.spines['top'].set_visible(False)
                    fig.tight_layout(); st.pyplot(fig); plt.close()

                    # Simulacro trend
                    sim_weeks = [(w["week"][:12], w["simulacro"]) for w in weeks if w["simulacro"]]
                    if len(sim_weeks) >= 2:
                        fig, ax = plt.subplots(figsize=(10, 2.5)); fig.patch.set_alpha(0); ax.set_facecolor('none')
                        ax.plot([s[0] for s in sim_weeks], [s[1] for s in sim_weeks], 'o-', color='#0F2B3D', linewidth=2, markersize=8)
                        for i, (lbl, sc) in enumerate(sim_weeks):
                            ax.text(i, sc + 3, str(sc), ha='center', fontsize=10, color='#0F2B3D', fontweight='500')
                        ax.set_ylabel('Score', fontsize=10, color='#5a6a7e')
                        ax.set_title('Simulacro progression', fontsize=12, color='#0F2B3D', fontweight='500', loc='left')
                        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                        plt.xticks(rotation=45, ha='right', fontsize=8)
                        fig.tight_layout(); st.pyplot(fig); plt.close()

                # Recent entries
                for w in reversed(weeks[-5:]):
                    energy_colors = ['#a32d2d','#ba7517','#6b7a8d','#0f6e56','#085041']
                    energy_labels = ['Very low','Low','Okay','Good','Great']
                    e_color = energy_colors[w["energy"]]
                    sim_text = f' | Simulacro: {w["simulacro"]}' if w["simulacro"] else ''
                    st.markdown(f"""<div style="padding:12px 16px;margin-bottom:6px;background:#fff;border:1px solid #E5DCC5;border-radius:8px;">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                            <span style="font-size:13px;font-weight:500;color:#0F2B3D;">{w["week"]}</span>
                            <span style="font-size:11px;color:{e_color};font-weight:500;">{energy_labels[w["energy"]]}</span>
                        </div>
                        <p style="font-size:12px;color:#6b7a8d;margin:0;">{w["hours"]}h studied{sim_text} | {', '.join(w["subjects"][:4]) if w["subjects"] else 'No subjects logged'}</p>
                        {"<p style='font-size:12px;color:#3a4a5e;margin:4px 0 0;font-style:italic;'>" + w["notes"][:120] + "</p>" if w["notes"] else ""}
                    </div>""", unsafe_allow_html=True)

                if st.button("Clear all data", key="mir_clear"):
                    st.session_state.mir_plan = {"weeks": [], "subjects": {}}
                    if mir_user: save_user_data(mir_user, "mir_plan", st.session_state.mir_plan)
                    st.rerun()

        with plan_tab2:
            subjects = st.session_state.mir_plan.get("subjects", {})
            if not subjects:
                st.markdown('<div class="content-card" style="text-align:center;margin-top:12px;"><p style="color:#6b7a8d;">No subjects logged yet. Log your first week to start tracking coverage.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown("""<div style="margin-top:12px;">
                    <h4 style="font-family:'Fraunces',serif;font-size:15px;color:#0F2B3D;margin-bottom:12px;">Subject coverage</h4>
                </div>""", unsafe_allow_html=True)
                sorted_subjects = sorted(subjects.items(), key=lambda x: x[1], reverse=True)
                max_count = sorted_subjects[0][1] if sorted_subjects else 1
                for subj, count in sorted_subjects:
                    width = count / max_count * 100
                    st.markdown(f"""<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                        <span style="font-size:13px;color:#0F2B3D;min-width:160px;font-weight:500;">{subj}</span>
                        <div style="flex:1;height:8px;background:#eef1f5;border-radius:4px;overflow:hidden;">
                            <div style="height:100%;width:{width:.0f}%;background:#534ab7;border-radius:4px;"></div>
                        </div>
                        <span style="font-size:11px;color:#6b7a8d;min-width:50px;text-align:right;">{count} week{"s" if count != 1 else ""}</span>
                    </div>""", unsafe_allow_html=True)

                # Missing subjects checklist
                all_mir_subjects = ["Cardiology","Pneumology","Nephrology","Digestive","Endocrinology","Haematology",
                    "Rheumatology","Neurology","Infectious diseases","Dermatology","Ophthalmology","ENT",
                    "Traumatology","Urology","Gynaecology","Obstetrics","Paediatrics","Psychiatry",
                    "Surgery","Pharmacology","Biostatistics","Preventive medicine","Legal medicine"]
                covered = set(s.lower() for s in subjects.keys())
                missing = [s for s in all_mir_subjects if s.lower() not in covered]
                if missing:
                    st.markdown(f"""<div class="content-card" style="margin-top:16px;border-left:3px solid #ef9f27;">
                        <h4>Not yet covered ({len(missing)} subjects)</h4>
                        <p style="color:#5a6a7e;">{', '.join(missing)}</p>
                    </div>""", unsafe_allow_html=True)

    # ═══ TAB 5: MIR STORIES ═══
    with tabs[4]:
        st.markdown("""<div style="margin-top:12px;margin-bottom:16px;">
            <p style="font-size:13px;color:#6b7a8d;">From students who've been through it. Composite narratives — no individual is identified.</p>
        </div>""", unsafe_allow_html=True)

        mir_stories = [
            ("I scored 2,000 places lower than my best simulacro predicted. I spent two weeks in a fog of disappointment. Then I chose a specialty I'd never seriously considered — and three years later, it's the best decision I ever made. The number doesn't define your career.",
             "Matched into a 'second-choice' specialty that turned out to be a better personality fit than the original target."),
            ("The worst part wasn't the studying — it was the loneliness. By month 4, I'd stopped answering messages. A friend showed up at my door and said 'we're going for a walk.' That walk probably saved my preparation. After restructuring to include daily walks and weekly dinners, my simulacro scores actually improved.",
             "Rest and social connection aren't enemies of performance. They're part of it."),
            ("I did the academia route and the competitive atmosphere nearly broke me. Everyone comparing scores, performing confidence they didn't feel. I started lying about my results because the real ones made me feel like a failure. When I finally told a friend the truth, she said 'me too.'",
             "Switched to home study with two partners. Reduced comparison, maintained accountability. Matched into preferred specialty."),
            ("I took a gap year before the MIR because I was burned out from 6th year. Best decision of my life. When I sat down to study, I actually wanted to learn. My score was higher than classmates who went straight in exhausted.",
             "Gap years carry no stigma in the matching process. Students who rest often perform better."),
            ("The eleccion was more stressful than the exam itself. Watching numbers tick down, calculating in real time. My hands were shaking. I wish someone had told me: have a Plan A, B, and C, and make sure you'd be genuinely okay with all three.",
             "Research your top 3 specialties and top 3 cities thoroughly, so any combination feels like a win."),
        ]

        for quote, outcome in mir_stories:
            st.markdown(f"""<div class="story-card">
                <div class="quote">"{quote}"</div>
            </div>""", unsafe_allow_html=True)
            with st.expander("What happened"):
                st.markdown(f'<p style="font-size:13px;color:#3a4a5e;line-height:1.6;">{outcome}</p>', unsafe_allow_html=True)

    # ═══ TAB 6: POST-MIR ═══
    with tabs[5]:
        st.markdown("""<div class="content-card" style="margin-top:12px;">
            <h4>The gap nobody talks about</h4>
            <p>Between the exam and residency, there's a period of profound transition. Relief, emptiness, results anxiety, eleccion stress, relocation, imposter syndrome — often all at once. This is normal.</p>
        </div>""", unsafe_allow_html=True)

        transitions = [
            ("Waiting for results", "Limit score-estimation conversations to once per day. Unfollow MIR anxiety groups on social media. Fill the time with things you postponed."),
            ("Results day", "Whatever you score, give yourself 48 hours before any decisions. High scorers often feel unexpectedly empty. Lower-than-expected scorers need time before concluding anything."),
            ("The eleccion", "Prepare 3 viable specialty-location combinations. Research each genuinely. On the day, trust your preparation."),
            ("Relocation", "Give yourself permission to not love the new city immediately. Find one social anchor quickly: a gym, a colleague, a coffee shop you like."),
            ("First weeks of residency", "Imposter syndrome is nearly universal. You will feel incompetent. This is learning, not inadequacy. Ask questions freely."),
            ("Long-term", "PAIME and 024 remain available during residency. Burnout risk actually increases in the first year. If you were flagged during screening, monitor those same dimensions."),
        ]

        for title, text in transitions:
            st.markdown(f"""<div style="display:flex;gap:14px;margin:12px 0;">
                <div style="min-width:4px;background:#3d8baa;border-radius:2px;"></div>
                <div>
                    <h4 style="font-family:'Fraunces',serif;font-size:15px;font-weight:500;color:#0F2B3D;margin:0 0 4px;">{title}</h4>
                    <p style="font-size:13px;color:#3a4a5e;line-height:1.65;margin:0;">{text}</p>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="content-card" style="text-align:center;margin-top:16px;">
            <p style="font-family:'Fraunces',serif;font-size:15px;color:#0F2B3D;line-height:1.7;">The MIR is one day. Your career is 40 years.<br>Take care of the person who has to live both.</p>
            <p style="margin-top:12px;">
                <a href="tel:024" style="display:inline-block;background:#0F2B3D;color:#fff;padding:9px 24px;border-radius:8px;text-decoration:none;font-weight:500;font-size:13px;">024</a>
                &nbsp;&nbsp;
                <a href="https://www.icomem.es/seccion/SALUD-MENTAL-MEDICO" target="_blank" style="display:inline-block;background:#3d8baa;color:#fff;padding:9px 24px;border-radius:8px;text-decoration:none;font-weight:500;font-size:13px;">PAIME</a>
            </p>
        </div>""", unsafe_allow_html=True)

    nav_back_to_hub()


# ============================================================================
# MIR: INSTITUTION PAGE
# ============================================================================

def page_mir_institution():
    render_top_bar()
    render_breadcrumb(("MedMind","gateway"),("Institution hub","institution_hub"),"MIR support protocol")

    st.markdown("""<div style="padding:16px 0 8px;">
        <h3 style="font-size:20px;font-weight:600;color:#0F2B3D;">MIR preparation — institutional support protocol</h3>
        <p style="font-size:13px;color:#6b7a8d;">Framework for supporting students through the highest-pressure period in Spanish medical training.</p>
    </div>""", unsafe_allow_html=True)

    tabs = st.tabs(["📊 Stress impact data", "🛡️ Support protocol", "📈 Cohort tracking"])

    # ═══ TAB 1: STRESS IMPACT DATA ═══
    with tabs[0]:
        st.markdown('<p class="section-header">MIR preparation stress: the evidence</p>', unsafe_allow_html=True)

        st.markdown("""<div class="content-card">
            <h4>Why MIR preparation is a distinct mental health risk</h4>
            <p>The MIR is not simply another exam. It is a single, high-stakes assessment that determines specialty assignment, geographic placement, and career trajectory. Preparation typically involves 8-12 months of intensive, largely solitary study — often outside university structures, without access to faculty support, peer networks, or institutional welfare services. This creates a unique risk profile:</p>
        </div>""", unsafe_allow_html=True)

        risk_factors = [
            ("🏠 Social isolation", "Students leave the daily social fabric of the faculty. Many study alone at home or in academias where interaction is competitive rather than supportive. DABE data shows social support is a top protective factor — and MIR preparation systematically erodes it."),
            ("🏆 Ranking as identity", "Unlike pass/fail exams, the MIR produces a public ranking. Students stop being 'good doctors in training' and become 'number 3,847 out of 12,000.' This externalises self-worth and creates a zero-sum mindset where every other student is a competitor."),
            ("📅 Extended duration", "Exam stress is typically acute (1-2 weeks). MIR stress is chronic (8-12 months). Chronic stress has fundamentally different neurological and psychological effects, including cortisol dysregulation, impaired memory consolidation, and cumulative burnout."),
            ("🎓 Post-graduation vulnerability", "Many students prepare for the MIR after graduating, meaning they lose access to university counselling services precisely when they need them most. They are no longer students but not yet residents — an institutional blind spot."),
            ("⚖️ All-or-nothing framing", "The cultural narrative around the MIR frames it as a single defining moment. Students who score below expectations often experience grief-like reactions. Students who take gap years face social stigma. This framing is disproportionate and harmful."),
        ]

        for title, desc in risk_factors:
            st.markdown(f'<div class="content-card"><h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

        # Compounding risk visual
        st.markdown('<p class="section-header">Compounding vulnerability</p>', unsafe_allow_html=True)
        st.markdown("""<div class="content-card" style="border-left:4px solid #e24b4a;">
            <h4>Students at highest risk during MIR preparation</h4>
            <p>DABE screening data can identify students who enter the MIR period already vulnerable. The following profiles are at compounded risk:</p>
            <p style="margin-top:10px;">
            • Students with <strong>moderate-to-high burnout</strong> at Year 5/6 screening (36.8% of cohort) entering sustained intensive study<br>
            • Students with <strong>low social support</strong> (bottom quartile) losing their remaining social structure<br>
            • Students with <strong>prior depressive episodes</strong> facing 8+ months of isolation and pressure<br>
            • <strong>Non-heterosexual students</strong> (identified as higher SI risk in SHAP) losing faculty-based support networks<br>
            • <strong>First-generation medical students</strong> who may lack family understanding of the MIR process</p>
            <p style="margin-top:10px;font-weight:500;">Screening in Year 5/6 should explicitly flag these compounding risks and trigger proactive outreach before the MIR period begins.</p>
        </div>""", unsafe_allow_html=True)

    # ═══ TAB 2: SUPPORT PROTOCOL ═══
    with tabs[1]:
        st.markdown('<p class="section-header">Institutional support recommendations</p>', unsafe_allow_html=True)

        protocols = [
            ("01", "Maintain welfare access post-graduation", "Students preparing for the MIR after graduating lose access to university counselling. Recommendation: extend welfare service eligibility for 18 months post-graduation, or until residency start. The cost is minimal; the impact is significant for the ~15% who need it."),
            ("02", "Pre-MIR wellbeing screening", "Offer a targeted screening at the start of the MIR preparation period (typically September-October). Use MedMind's 35-item screening to identify students entering preparation with existing vulnerabilities. Flag for proactive outreach."),
            ("03", "Facilitated MIR study groups", "Organise optional faculty-supported study groups (4-6 students) that meet weekly. The primary purpose is social connection, not academic content. Isolation is the single biggest wellbeing risk during MIR preparation — structured groups directly counter it."),
            ("04", "Periodic check-in programme", "Assign each MIR-preparing student a welfare contact (tutor, counsellor, or trained peer) who reaches out monthly. Brief, low-pressure: 'How are you doing? Is there anything you need?' Students rarely self-refer during MIR preparation — proactive contact is essential."),
            ("05", "Academia liaison", "If students attend external academias, establish a communication channel with major providers. Academias see students daily but have no welfare training. A basic agreement to flag concerning behaviour (e.g., sudden absence, visible distress) could enable early intervention."),
            ("06", "Gap year normalisation", "Publicly acknowledge that taking a gap year before the MIR is a legitimate and sometimes beneficial choice. Publish data showing that gap-year students perform comparably. Reduce the cultural stigma that forces exhausted students into immediate preparation."),
            ("07", "Post-MIR transition support", "Offer a structured transition programme for the period between results and residency start: logistics support for relocation, social events for matched cohorts, and a wellbeing check-in before residency begins. This period is psychologically underserved."),
            ("08", "Crisis visibility during preparation", "Ensure 024 and PAIME information reaches students during MIR preparation specifically — not just during faculty years. Consider a targeted communication in January-February (peak exam stress) with resources and normalising messaging."),
        ]

        for num, title, desc in protocols:
            st.markdown(f'<div class="content-card" style="border-left:4px solid #0F2B3D;"><div style="display:flex;align-items:flex-start;gap:16px;"><div style="min-width:44px;height:44px;background:#0F2B3D;color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:14px;">{num}</div><div><h4 style="margin-top:8px;">{title}</h4><p>{desc}</p></div></div></div>', unsafe_allow_html=True)

    # ═══ TAB 3: COHORT TRACKING ═══
    with tabs[2]:
        st.markdown('<p class="section-header">MIR cohort tracking framework</p>', unsafe_allow_html=True)
        st.markdown('<div class="content-card"><h4>Longitudinal monitoring through the MIR period</h4><p>Standard annual screening misses the MIR preparation period entirely. A dedicated tracking framework follows students through the most critical transition in their training.</p></div>', unsafe_allow_html=True)

        checkpoints = [
            ("T0", "September (prep start)", "Baseline screening using MedMind's 35-item tool. Establish each student's pre-MIR risk profile. Flag students with existing moderate/high risk for proactive support."),
            ("T1", "December (3 months in)", "Brief check-in (8 items from the self-monitoring tracker). Compare to T0 baseline. Identify students showing deterioration — especially on exhaustion, social connection, and sleep."),
            ("T2", "February (1 month pre-exam)", "Peak pressure assessment. Focus on anxiety, sleep quality, and coping. This is when crisis risk is highest. Ensure all students have 024/PAIME information fresh in mind."),
            ("T3", "Post-exam (2 weeks after)", "Decompression check-in. Screen for post-exam depression, emptiness, and relief-crash. Students often don't recognise that the 'flatness' they feel after the exam is a treatable condition."),
            ("T4", "Post-eleccion / pre-residency", "Transition readiness assessment. Screen for relocation anxiety, imposter syndrome anticipation, and unresolved MIR-related distress. Connect with receiving hospital's welfare services."),
        ]

        import matplotlib.pyplot as plt; import matplotlib; matplotlib.use('Agg')
        fig, ax = plt.subplots(figsize=(10, 2.5)); fig.patch.set_alpha(0); ax.set_facecolor('none')
        months = ['Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun']
        stress = [3, 4, 5, 6, 7, 8, 9, 10, 6, 4]
        ax.fill_between(range(len(months)), stress, alpha=0.15, color='#e24b4a')
        ax.plot(range(len(months)), stress, color='#e24b4a', linewidth=2.5)
        for i, (t, m) in enumerate([(0,'T0'),(3,'T1'),(5,'T2'),(7,'T3'),(9,'T4')]):
            ax.plot(t, stress[t], 'o', color='#0F2B3D', markersize=12, zorder=5)
            ax.text(t, stress[t]+0.6, m, ha='center', fontsize=11, fontweight='700', color='#0F2B3D')
        ax.set_xticks(range(len(months))); ax.set_xticklabels(months, fontsize=10)
        ax.set_ylabel('Stress level', fontsize=10, color='#5a6a7e'); ax.set_ylim(0, 12)
        ax.set_title('Estimated stress trajectory during MIR preparation', fontsize=13, color='#0F2B3D', fontweight='600')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close()

        for code, timing, desc in checkpoints:
            st.markdown(f'<div class="content-card" style="border-left:4px solid #48a8c9;"><div style="display:flex;gap:14px;"><div style="min-width:40px;height:40px;background:#48a8c9;color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;">{code}</div><div><h4>{timing}</h4><p>{desc}</p></div></div></div>', unsafe_allow_html=True)

        st.markdown("""<div class="content-card" style="border-left:4px solid #ef9f27;">
            <h4>⚠️ Implementation note</h4>
            <p>MIR cohort tracking requires institutional commitment to maintaining contact with students who may no longer be enrolled. This means: updated contact details before graduation, explicit consent for post-graduation follow-up, and a designated staff member responsible for MIR cohort welfare. The recommended staffing is 0.2 FTE per 100 MIR-preparing students.</p>
        </div>""", unsafe_allow_html=True)

    nav_back_to_hub()


# ============================================================================
# MAIN ROUTING
# ============================================================================
# NEW: INSTITUTIONAL CSV UPLOAD
# ============================================================================

def page_csv_upload():
    render_top_bar()
    render_breadcrumb(("MedMind","gateway"),("Institution hub","institution_hub"),"Data upload")

    st.markdown("""<div style="padding:16px 0 8px;">
        <h3 style="font-family:'Fraunces',serif;font-size:22px;font-weight:600;color:#0F2B3D;">Upload your screening data</h3>
        <p style="font-size:14px;color:#888780;">Upload anonymised CSV data from your own screening programme to generate the same visualisations as the simulated dashboard.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="content-card"><h4>Required format</h4>
    <p>CSV with columns: <code>student_id</code>, <code>course_year</code> (1-6), <code>gender</code> (M/F), <code>depression_risk</code> (0-1), <code>burnout_risk</code> (0-1), <code>anxiety_risk</code> (0-1), <code>referral_status</code> (none/recommended/active). Optional: <code>timestamp</code>, <code>faculty</code>.</p>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="inst_csv")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df)} records")
            import matplotlib.pyplot as plt; import matplotlib; matplotlib.use('Agg')

            required = ['course_year', 'depression_risk', 'burnout_risk', 'anxiety_risk']
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
                nav_back_to_hub(); return

            # Summary metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.markdown(f'<div class="metric-card"><div class="m-value">{len(df)}</div><div class="m-label">Students screened</div></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="metric-card"><div class="m-value">{(df.depression_risk>=0.3).mean()*100:.0f}%</div><div class="m-label">Depression risk</div></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="metric-card"><div class="m-value">{(df.burnout_risk>=0.3).mean()*100:.0f}%</div><div class="m-label">Burnout risk</div></div>', unsafe_allow_html=True)
            with c4: st.markdown(f'<div class="metric-card"><div class="m-value">{(df.anxiety_risk>=0.3).mean()*100:.0f}%</div><div class="m-label">Anxiety risk</div></div>', unsafe_allow_html=True)

            # Risk by year
            st.markdown('<p class="section-header">Risk prevalence by year</p>', unsafe_allow_html=True)
            if 'course_year' in df.columns:
                by_year = df.groupby('course_year')[['depression_risk','burnout_risk','anxiety_risk']].apply(lambda x: (x>=0.3).mean()*100)
                fig, ax = plt.subplots(figsize=(10,4)); fig.patch.set_alpha(0); ax.set_facecolor('none')
                years = by_year.index.values
                w = 0.25
                ax.bar(years-w, by_year['depression_risk'], w, color='#48A8C9', label='Depression')
                ax.bar(years, by_year['burnout_risk'], w, color='#EF9F27', label='Burnout')
                ax.bar(years+w, by_year['anxiety_risk'], w, color='#0F6E56', label='Anxiety')
                ax.set_xlabel('Course year'); ax.set_ylabel('% at risk'); ax.legend()
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                fig.tight_layout(); st.pyplot(fig); plt.close()

            # Referral funnel
            if 'referral_status' in df.columns:
                st.markdown('<p class="section-header">Referral funnel</p>', unsafe_allow_html=True)
                funnel = df.referral_status.value_counts()
                for status, count in funnel.items():
                    pct = count/len(df)*100
                    st.markdown(f'<div style="margin-bottom:6px;"><span style="font-size:14px;font-weight:500;color:#0F2B3D;">{status.title()}</span><span style="font-size:14px;color:#888780;margin-left:8px;">{count} ({pct:.0f}%)</span><div style="height:8px;background:#eef1f5;border-radius:4px;overflow:hidden;margin-top:4px;"><div style="height:100%;width:{pct:.0f}%;background:#48A8C9;border-radius:4px;"></div></div></div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing CSV: {e}")

    nav_back_to_hub()


# ============================================================================
# NEW: INTERVENTION PRIORITY RECOMMENDER
# ============================================================================

def page_intervention_recommender():
    render_top_bar()
    render_breadcrumb(("MedMind","gateway"),("Institution hub","institution_hub"),"Intervention recommender")

    st.markdown("""<div style="padding:16px 0 8px;">
        <h3 style="font-family:'Fraunces',serif;font-size:22px;font-weight:600;color:#0F2B3D;">Intervention priority recommender</h3>
        <p style="font-size:14px;color:#888780;">Input your data. Get a ranked, costed action plan backed by the SHAP analysis and DABE benchmarks.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-header">Your university profile</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        n_students = st.number_input("Medical students (total)", 100, 3000, 600, step=50, key="ir_n")
        dep_rate = st.slider("Depression prevalence (%)", 5, 60, 25, key="ir_dep")
        burn_rate = st.slider("Burnout prevalence (%)", 10, 70, 35, key="ir_burn")
        anx_rate = st.slider("Anxiety prevalence (%)", 5, 50, 20, key="ir_anx")
    with c2:
        cost_per_student = st.number_input("Cost per student per year (EUR)", 1000, 30000, 8000, step=500, key="ir_cost")
        low_support = st.slider("Low social support (%)", 5, 50, 20, key="ir_support")
        low_satisfaction = st.slider("Dissatisfied with studies (%)", 5, 40, 15, key="ir_sat")
        dropout_rate = st.slider("Estimated dropout/repeat rate among at-risk (%)", 5, 30, 15, key="ir_drop")

    if st.button("Generate action plan", key="ir_go", use_container_width=True):

        # Compute baseline cost of inaction
        avg_risk = (dep_rate + burn_rate + anx_rate) / 3 / 100
        at_risk = int(n_students * avg_risk)
        dropouts = int(at_risk * dropout_rate / 100)
        annual_loss = dropouts * cost_per_student
        st.session_state["ir_annual_loss"] = annual_loss

        # Summary metrics
        st.markdown('<p class="section-header">Your baseline</p>', unsafe_allow_html=True)
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1: st.markdown(f'<div class="metric-card"><div class="m-value">{at_risk}</div><div class="m-label">Students at risk</div></div>', unsafe_allow_html=True)
        with mc2: st.markdown(f'<div class="metric-card"><div class="m-value">{dropouts}</div><div class="m-label">Estimated dropouts/yr</div></div>', unsafe_allow_html=True)
        with mc3: st.markdown(f'<div class="metric-card"><div class="m-value">{chr(8364)}{annual_loss:,.0f}</div><div class="m-label">Annual cost of inaction</div></div>', unsafe_allow_html=True)
        with mc4: st.markdown(f'<div class="metric-card"><div class="m-value">{chr(8364)}{cost_per_student:,.0f}</div><div class="m-label">Cost per student</div></div>', unsafe_allow_html=True)

        # Build intervention list with concrete data
        interventions = []

        # 1. Peer mentorship (always high impact due to SHAP)
        mentor_cost = int(n_students * 3.5)  # ~€3.50/student for coordination
        mentor_prevented = max(1, int(dropouts * 0.20))  # 20% of dropouts prevented
        mentor_saved = mentor_prevented * cost_per_student
        interventions.append({
            "name": "Peer mentorship programme",
            "score": low_support * 2.5 + 20,
            "annual_cost": mentor_cost,
            "students_helped": mentor_prevented,
            "savings": mentor_saved,
            "roi": mentor_saved / max(mentor_cost, 1),
            "timeline": "2-3 months to launch",
            "evidence": f"Social support is the #1 protective factor across all 4 outcomes (SHAP). Your {low_support}% low-support rate {'exceeds' if low_support > 20 else 'is near'} the DABE average (18%). Depression prevalence drops from 64% to 14% between students with 0 vs 5+ support persons.",
            "actions": "Recruit Year 4-6 volunteer mentors. Match 1:3 with Years 1-2. Monthly structured meetings + ad-hoc contact. Train mentors in active listening (one 3-hour session).",
        })

        # 2. Proactive screening
        screen_cost = int(n_students * 12)  # ~€12/student for platform + admin
        screen_prevented = max(1, int(dropouts * 0.25))
        screen_saved = screen_prevented * cost_per_student
        interventions.append({
            "name": "Systematic screening programme",
            "score": dep_rate * 2.2 + 15,
            "annual_cost": screen_cost,
            "students_helped": screen_prevented,
            "savings": screen_saved,
            "roi": screen_saved / max(screen_cost, 1),
            "timeline": "1-2 months (using MedMind)",
            "evidence": f"Depression prediction AUC = 0.947. At 90% recall threshold, 9 of 10 at-risk students detected with 62% precision. Current self-referral rates in Spanish universities are below 30%. Your {dep_rate}% depression rate {'exceeds' if dep_rate > 25 else 'is near'} the DABE average (23%).",
            "actions": "Deploy 35-item screening at semester start. Automated risk triage (green/amber/red). Referral pathways to counselling for amber/red. Repeat at semester end for longitudinal tracking.",
        })

        # 3. Protected rest / curriculum reform
        rest_cost = int(n_students * 0.5)  # Minimal direct cost — policy change
        rest_prevented = max(1, int(dropouts * 0.15))
        rest_saved = rest_prevented * cost_per_student
        interventions.append({
            "name": "Protected rest and workload reform",
            "score": burn_rate * 2.0 + 10,
            "annual_cost": rest_cost,
            "students_helped": rest_prevented,
            "savings": rest_saved,
            "roi": rest_saved / max(rest_cost, 1),
            "timeline": "6-12 months (curriculum committee)",
            "evidence": f"Burnout nearly doubles from Year 1 (22.6%) to Year 6 (44.8%) in DABE data. SHAP identifies effort-grade mismatch and academic demands as top burnout drivers. Your {burn_rate}% burnout rate {'exceeds' if burn_rate > 35 else 'is near'} the DABE average (33%).",
            "actions": "Mandate one rest day per week. Cap clinical shift length at 12 hours. Annual workload audit against WFME recommendations. Introduce 2-3 wellbeing days per semester.",
        })

        # 4. Career counselling
        career_cost = int(n_students * 8)  # ~€8/student
        career_prevented = max(1, int(dropouts * 0.10))
        career_saved = career_prevented * cost_per_student
        interventions.append({
            "name": "Career counselling service",
            "score": low_satisfaction * 1.8 + 10,
            "annual_cost": career_cost,
            "students_helped": career_prevented,
            "savings": career_saved,
            "roi": career_saved / max(career_cost, 1),
            "timeline": "3-4 months",
            "evidence": f"Low satisfaction with choosing medicine is uniquely predictive of suicidal ideation (SHAP) — it does not appear as a risk factor for the other 3 targets. This makes vocational misalignment a specific warning signal. Your {low_satisfaction}% dissatisfaction rate warrants targeted intervention.",
            "actions": "Dedicated career advisory service (not generic university careers). Focus on Years 3-4 where dissatisfaction peaks. Include specialty exploration workshops and shadowing programmes.",
        })

        # 5. Anxiety workshops
        anx_cost = int(min(n_students * anx_rate / 100, 80) * 150)  # ~€150/participant for 6-session CBT
        anx_prevented = max(1, int(dropouts * 0.12))
        anx_saved = anx_prevented * cost_per_student
        interventions.append({
            "name": "Anxiety management workshops",
            "score": anx_rate * 1.5 + 5,
            "annual_cost": anx_cost,
            "students_helped": anx_prevented,
            "savings": anx_saved,
            "roi": anx_saved / max(anx_cost, 1),
            "timeline": "2-3 months",
            "evidence": f"Anxiety responds well to structured CBT workshops (effect sizes 0.5-0.8). SHAP shows exhaustion and low social support as top anxiety drivers — workshops address both through skills training and group format. Your {anx_rate}% anxiety rate {'exceeds' if anx_rate > 20 else 'is near'} the DABE average (18%).",
            "actions": "6-session CBT-based programme before exam periods. Include 4-7-8 breathing, cognitive defusion, study scheduling. Group format (8-12 students) provides social benefit alongside skills.",
        })

        # 6. LGBTQ+ outreach (always included)
        lgbtq_cost = int(n_students * 1.5)
        interventions.append({
            "name": "LGBTQ+ inclusive outreach",
            "score": 30,
            "annual_cost": lgbtq_cost,
            "students_helped": "—",
            "savings": "—",
            "roi": "—",
            "timeline": "1-2 months",
            "evidence": "Non-heterosexual orientation is the 2nd strongest predictor of suicidal ideation (SHAP). This is an ethical imperative regardless of prevalence data. Active visibility reduces minority stress.",
            "actions": "Visible LGBTQ+ affirming policies. Named ally staff. Ensure screening outreach explicitly reaches LGBTQ+ students. Review curriculum for heteronormative assumptions.",
        })

        # Sort by score
        interventions.sort(key=lambda x: x["score"], reverse=True)

        # Total investment and return
        total_cost = sum(i["annual_cost"] for i in interventions)
        total_saved = sum(i["savings"] for i in interventions if isinstance(i["savings"], (int, float)))
        total_prevented = sum(i["students_helped"] for i in interventions if isinstance(i["students_helped"], (int, float)))

        st.markdown('<p class="section-header">Recommended interventions — ranked by expected impact</p>', unsafe_allow_html=True)

        # Summary ROI card
        st.markdown(f"""<div style="background:linear-gradient(135deg,#fff,#F5F9FE);border:1px solid #48A8C9;border-radius:14px;padding:24px;margin-bottom:16px;">
            <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:20px;">
                <div><span style="font-size:12px;color:#888780;">Total annual investment</span><br><span style="font-family:'Fraunces',serif;font-size:24px;font-weight:600;color:#48A8C9;">{chr(8364)}{total_cost:,.0f}</span></div>
                <div><span style="font-size:12px;color:#888780;">Estimated annual savings</span><br><span style="font-family:'Fraunces',serif;font-size:24px;font-weight:600;color:#0F6E56;">{chr(8364)}{total_saved:,.0f}</span></div>
                <div><span style="font-size:12px;color:#888780;">Estimated dropouts prevented</span><br><span style="font-family:'Fraunces',serif;font-size:24px;font-weight:600;color:#0F2B3D;">{total_prevented}</span></div>
                <div><span style="font-size:12px;color:#888780;">Return on investment</span><br><span style="font-family:'Fraunces',serif;font-size:24px;font-weight:600;color:#48A8C9;">{total_saved/max(total_cost,1):.0f}x</span></div>
            </div>
        </div>""", unsafe_allow_html=True)

        for rank, iv in enumerate(interventions, 1):
            clr = "#48A8C9" if rank <= 2 else "#0F6E56" if rank <= 4 else "#888780"
            roi_text = f"{iv['roi']:.0f}x return" if isinstance(iv['roi'], (int, float)) else "Ethical imperative"
            cost_text = f"{chr(8364)}{iv['annual_cost']:,.0f}/year"
            saved_text = f"{chr(8364)}{iv['savings']:,.0f} saved" if isinstance(iv['savings'], (int, float)) else "Unquantifiable"

            st.markdown(f"""<div style="padding:20px 24px;margin-bottom:12px;background:#fff;border:1px solid #E5DCC5;border-radius:14px;border-left:4px solid {clr};">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
                    <div style="display:flex;align-items:center;gap:10px;">
                        <span style="font-size:11px;font-weight:600;color:#fff;background:{clr};padding:3px 12px;border-radius:10px;">#{rank}</span>
                        <span style="font-family:'Fraunces',serif;font-size:17px;font-weight:500;color:#0F2B3D;">{iv['name']}</span>
                    </div>
                    <span style="font-size:12px;font-weight:500;color:{clr};">{roi_text}</span>
                </div>
                <div style="display:flex;gap:24px;margin-bottom:12px;flex-wrap:wrap;">
                    <span style="font-size:13px;color:#5a6a7e;">Cost: <strong>{cost_text}</strong></span>
                    <span style="font-size:13px;color:#5a6a7e;">Savings: <strong>{saved_text}</strong></span>
                    <span style="font-size:13px;color:#5a6a7e;">Timeline: <strong>{iv['timeline']}</strong></span>
                </div>
                <p style="font-size:14px;color:#3a4a5e;line-height:1.65;margin-bottom:8px;">{iv['evidence']}</p>
                <details><summary style="font-size:13px;font-weight:500;color:#48A8C9;cursor:pointer;">Implementation steps</summary>
                <p style="font-size:13px;color:#3a4a5e;line-height:1.6;margin-top:8px;">{iv['actions']}</p></details>
            </div>""", unsafe_allow_html=True)

        # Connection to other tools
        st.markdown('<p class="section-header">Next steps</p>', unsafe_allow_html=True)
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.markdown('<div class="content-card" style="text-align:center;"><h4>Cost calculator</h4><p>Model the full financial case with detailed assumptions.</p></div>', unsafe_allow_html=True)
            if st.button("Open calculator", key="ir_calc", use_container_width=True):
                st.session_state.page = "calculator"; st.rerun()
        with cc2:
            st.markdown('<div class="content-card" style="text-align:center;"><h4>Policy brief</h4><p>Generate a one-page summary for your dean.</p></div>', unsafe_allow_html=True)
            if st.button("Generate brief", key="ir_policy", use_container_width=True):
                st.session_state.page = "policy_brief"; st.rerun()
        with cc3:
            st.markdown('<div class="content-card" style="text-align:center;"><h4>Implementation guide</h4><p>Detailed framework for early warning systems.</p></div>', unsafe_allow_html=True)
            if st.button("Read guide", key="ir_ew", use_container_width=True):
                st.session_state.page = "earlywarning"; st.rerun()

    nav_back_to_hub()


# ============================================================================
# NEW: DATA COLLECTION MODE (for external validation)
# ============================================================================

def page_data_collection():
    render_top_bar()
    render_breadcrumb(("MedMind","gateway"),("Institution hub","institution_hub"),"Data collection framework")

    st.markdown("""<div style="padding:16px 0 8px;">
        <h3 style="font-family:'Fraunces',serif;font-size:22px;font-weight:600;color:#0F2B3D;">Data collection framework</h3>
        <p style="font-size:14px;color:#888780;">A standardised protocol for universities to collect screening data that contributes to model validation and improvement.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="content-card"><h4>Why this matters</h4>
    <p>The current models were trained on DABE 2020 data (pre-pandemic). External validation on independent samples
    is the most important next step for establishing clinical reliability. Every university that collects data using
    this standardised protocol contributes to that validation — and gets its own institutional insights in return.</p></div>""", unsafe_allow_html=True)

    stages = [
        ("Ethics and approval", "Submit protocol to your institutional ethics board. Template protocol available below. Key requirements: informed consent, anonymisation at source, GDPR compliance, opt-out mechanism."),
        ("Questionnaire deployment", "Use the same 35-item screening questionnaire as MedMind's screening model (Model B). This uses only demographic, academic, social, and lifestyle items — no clinical instruments required. Administer at semester start."),
        ("Data format", "Export as CSV matching the MedMind schema: student_id (anonymised), course_year, gender, age, and all 35 screening features. Feature names must match exactly for model compatibility."),
        ("Upload and analysis", "Use the CSV upload tool (Institution hub) to generate your institutional dashboard. Data stays on your servers — MedMind processes it locally, nothing is transmitted."),
        ("Validation contribution", "If you wish to contribute anonymised data to the validation pool, contact the research team. Pooled multi-site data enables cross-validation and generalisability testing."),
    ]

    for i, (title, desc) in enumerate(stages, 1):
        st.markdown(f"""<div style="display:flex;gap:16px;margin-bottom:14px;">
            <div style="width:32px;height:32px;border-radius:50%;background:#D6EDF2;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:14px;font-weight:600;color:#48A8C9;">{i}</div>
            <div>
                <h4 style="font-family:'Fraunces',serif;font-size:16px;font-weight:500;color:#0F2B3D;margin-bottom:4px;">{title}</h4>
                <p style="font-size:14px;color:#3a4a5e;line-height:1.65;">{desc}</p>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="content-card" style="border-left:4px solid #48A8C9;">
        <h4>Technical specifications</h4>
        <p>The screening model (Model B) uses 35 features covering demographics (age, gender, course year, sexual orientation),
        academic variables (satisfaction, attendance, effort-grade alignment), social support (4 Duke-UNC items),
        life events (5 indicators), substance use (tobacco, alcohol, cannabis), perceived difficulties (12 problem areas),
        and psychiatric medication use. Full feature list and encoding documentation available in the GitHub repository.</p>
    </div>""", unsafe_allow_html=True)

    nav_back_to_hub()


# ============================================================================
# NEW: WELLBEING OFFICER TRIAGE DASHBOARD
# ============================================================================

def _seed_triage_cases():
    """Seed simulated cases on first visit. Uses pseudonymous IDs only."""
    import random
    from datetime import datetime, timedelta
    random.seed(42)
    outcomes = ["Depression", "Burnout", "Anxiety"]
    years = [1, 2, 3, 4, 5, 6]
    cases = []
    for i in range(14):
        days_ago = random.randint(0, 21)
        risk_prob = round(random.uniform(0.35, 0.92), 2)
        level = "High" if risk_prob >= 0.65 else "Moderate"
        cases.append({
            "id": f"MS-{1000 + i}",
            "year": random.choice(years),
            "outcome": random.choice(outcomes),
            "risk": risk_prob,
            "level": level,
            "flagged_date": (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d"),
            "days_ago": days_ago,
            "status": "new" if days_ago < 3 else random.choice(["new", "contacted", "in_progress", "resolved"]),
            "notes": [],
        })
    return cases

def page_triage_dashboard():
    render_top_bar()
    render_breadcrumb(("MedMind","gateway"),("Institution hub","institution_hub"),"Wellbeing triage")

    st.markdown("""<div style="padding:16px 0 8px;">
        <h3 style="font-family:'Fraunces',serif;font-size:22px;font-weight:600;color:#0F2B3D;">Wellbeing officer triage dashboard</h3>
        <p style="font-size:14px;color:#888780;">Incoming red flags from screening. Workflow: <strong>new → contacted → in progress → resolved</strong>. Students are identified by anonymised IDs only.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div style="background:#FAEEDA;border:1px solid #FAC775;border-radius:12px;padding:14px 18px;margin-bottom:16px;">
        <p style="font-size:13px;color:#854F0B;margin:0;"><strong>Demonstration mode.</strong> The cases below are simulated for illustration. In a real deployment, they would be drawn from live screening results with clinician oversight and institutional protocols.</p>
    </div>""", unsafe_allow_html=True)

    if "triage_cases" not in st.session_state:
        st.session_state.triage_cases = _seed_triage_cases()

    cases = st.session_state.triage_cases

    # Metrics
    n_new = sum(1 for c in cases if c["status"] == "new")
    n_contacted = sum(1 for c in cases if c["status"] == "contacted")
    n_progress = sum(1 for c in cases if c["status"] == "in_progress")
    n_resolved = sum(1 for c in cases if c["status"] == "resolved")
    open_cases = n_new + n_contacted + n_progress

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1: st.markdown(f'<div class="metric-card" style="border-left:3px solid #c92f2f;"><div class="m-value">{n_new}</div><div class="m-label">New</div></div>', unsafe_allow_html=True)
    with mc2: st.markdown(f'<div class="metric-card" style="border-left:3px solid #c78017;"><div class="m-value">{n_contacted}</div><div class="m-label">Contacted</div></div>', unsafe_allow_html=True)
    with mc3: st.markdown(f'<div class="metric-card" style="border-left:3px solid #48A8C9;"><div class="m-value">{n_progress}</div><div class="m-label">In progress</div></div>', unsafe_allow_html=True)
    with mc4: st.markdown(f'<div class="metric-card" style="border-left:3px solid #0F6E56;"><div class="m-value">{n_resolved}</div><div class="m-label">Resolved</div></div>', unsafe_allow_html=True)

    # Filter
    st.markdown('<p class="section-header">Cases</p>', unsafe_allow_html=True)
    filter_opt = st.radio("Show", ["Open cases", "All", "New only", "Contacted", "In progress", "Resolved"], horizontal=True, key="triage_filter", label_visibility="collapsed")
    status_map = {"New only": ["new"], "Contacted": ["contacted"], "In progress": ["in_progress"], "Resolved": ["resolved"], "Open cases": ["new", "contacted", "in_progress"], "All": ["new", "contacted", "in_progress", "resolved"]}
    visible = [c for c in cases if c["status"] in status_map[filter_opt]]

    # Sort by priority: level + days_ago
    visible.sort(key=lambda c: (c["level"] != "High", c["days_ago"]))

    if not visible:
        st.markdown('<div class="content-card" style="text-align:center;"><p style="color:#888780;">No cases in this view.</p></div>', unsafe_allow_html=True)

    status_colors = {"new": "#c92f2f", "contacted": "#c78017", "in_progress": "#48A8C9", "resolved": "#0F6E56"}
    status_labels = {"new": "New", "contacted": "Contacted", "in_progress": "In progress", "resolved": "Resolved"}

    for c in visible:
        clr = status_colors[c["status"]]
        label = status_labels[c["status"]]
        level_clr = "#c92f2f" if c["level"] == "High" else "#c78017"

        with st.container():
            notes_html = f'<p style="font-size:12px;color:#5a6a7e;font-style:italic;margin:8px 0 0;">Note: {c["notes"][-1]}</p>' if c['notes'] else ''
            card_html = f'<div style="background:#fff;border:1px solid #E5DCC5;border-radius:12px;padding:18px 20px;margin-bottom:10px;border-left:4px solid {clr};"><div style="margin-bottom:6px;"><span style="font-family:\'Fraunces\',serif;font-size:15px;font-weight:500;color:#0F2B3D;">{c["id"]}</span>&nbsp;&nbsp;<span style="font-size:11px;font-weight:600;color:#fff;background:{clr};padding:2px 10px;border-radius:10px;">{label}</span>&nbsp;&nbsp;<span style="font-size:11px;font-weight:600;color:{level_clr};border:1px solid {level_clr};padding:2px 10px;border-radius:10px;">{c["level"]} risk</span></div><p style="font-size:13px;color:#3a4a5e;margin:0;">Year {c["year"]} · Flagged for {c["outcome"].lower()} · Risk score {c["risk"]:.0%} · Flagged {c["days_ago"]} days ago</p>{notes_html}</div>'
            st.markdown(card_html, unsafe_allow_html=True)

            # Workflow actions
            bc1, bc2, bc3, bc4 = st.columns(4)
            case_key = c["id"]
            with bc1:
                if c["status"] == "new":
                    if st.button("Mark contacted", key=f"tri_ct_{case_key}", use_container_width=True):
                        c["status"] = "contacted"; st.rerun()
            with bc2:
                if c["status"] == "contacted":
                    if st.button("Move to in progress", key=f"tri_prog_{case_key}", use_container_width=True):
                        c["status"] = "in_progress"; st.rerun()
            with bc3:
                if c["status"] in ["contacted", "in_progress"]:
                    if st.button("Mark resolved", key=f"tri_res_{case_key}", use_container_width=True):
                        c["status"] = "resolved"; st.rerun()
            with bc4:
                with st.expander("Add note", expanded=False):
                    note = st.text_input("Note", key=f"tri_note_{case_key}", label_visibility="collapsed", placeholder="Follow-up details...")
                    if st.button("Save note", key=f"tri_savenote_{case_key}", use_container_width=True):
                        if note.strip():
                            c["notes"].append(note.strip())
                            st.rerun()

    # Reset button for demo
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Reset demonstration data", key="triage_reset"):
        st.session_state.triage_cases = _seed_triage_cases(); st.rerun()

    nav_back_to_hub()


# ============================================================================
# MAIN ROUTING
# ============================================================================

def main():
    if "page" not in st.session_state: st.session_state.page = "gateway"
    if "role" not in st.session_state: st.session_state.role = ""

    routes = {
        "gateway": page_gateway, "student_hub": page_student_hub, "institution_hub": page_institution_hub,
        "screening": page_screening, "results": page_results, "agent": page_agent,
        "dashboard": page_dashboard, "historical": page_historical, "playbook": page_playbook,
        "shap_rankings": page_shap_rankings,
        "resources": page_resources, "coping": page_coping, "tracker": page_tracker,
        "earlywarning": page_earlywarning, "calculator": page_calculator, "stories": page_stories,
        # Extended features
        "action_plan": page_action_plan, "benchmark": page_benchmark, "exam_mode": page_exam_mode,
        "sleep_tracker": page_sleep_tracker, "letter": page_letter, "inst_benchmark": page_inst_benchmark,
        "evaluation": page_evaluation, "heatmap": page_heatmap, "policy_brief": page_policy_brief,
        "sim_dashboard": page_sim_dashboard,
        # MIR
        "mir_student": page_mir_student, "mir_institution": page_mir_institution,
        # Extended institutional features
        "csv_upload": page_csv_upload, "intervention_recommender": page_intervention_recommender,
        "data_collection": page_data_collection, "triage_dashboard": page_triage_dashboard,
    }
    routes.get(st.session_state.page, page_gateway)()

if __name__ == "__main__":
    main()
