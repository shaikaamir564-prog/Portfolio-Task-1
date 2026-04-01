import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Loan Prediction System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background: #0a0f1e;
    color: #e8eaf0;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1300px; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3.5rem 2rem 2.5rem;
    position: relative;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 300px;
    background: radial-gradient(ellipse, rgba(99,179,237,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,179,237,0.12);
    border: 1px solid rgba(99,179,237,0.3);
    color: #63b3ed;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.4rem;
    font-family: 'JetBrains Mono', monospace;
}
.hero h1 {
    font-size: clamp(2.4rem, 5vw, 3.8rem);
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #e8eaf0 30%, #63b3ed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.9rem;
}
.hero p {
    color: #8892a4;
    font-size: 1.05rem;
    font-weight: 300;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.7;
}

/* ── Section Headers ── */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 0.6rem;
}
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e8eaf0;
    margin-bottom: 1.4rem;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

/* ── Card ── */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(8px);
    transition: border-color 0.25s;
}
.card:hover { border-color: rgba(99,179,237,0.25); }

/* ── Streamlit widgets override ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {
    color: #8892a4 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}

div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-family: 'Sora', sans-serif !important;
}

/* Slider accent */
div[data-testid="stSlider"] > div > div > div > div {
    background: #63b3ed !important;
}

/* ── Button ── */
div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #2b6cb0, #4299e1);
    color: white;
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.04em;
    padding: 0.9rem 2rem;
    border: none;
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.2s;
    box-shadow: 0 4px 24px rgba(66,153,225,0.3);
    margin-top: 0.5rem;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #3182ce, #63b3ed);
    transform: translateY(-1px);
    box-shadow: 0 6px 32px rgba(66,153,225,0.45);
}

/* ── Result Card ── */
.result-approved {
    background: linear-gradient(135deg, rgba(72,187,120,0.12), rgba(56,161,105,0.06));
    border: 1.5px solid rgba(72,187,120,0.35);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-rejected {
    background: linear-gradient(135deg, rgba(252,129,74,0.12), rgba(229,62,62,0.06));
    border: 1.5px solid rgba(252,129,74,0.35);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-icon { font-size: 3.5rem; margin-bottom: 0.8rem; }
.result-verdict {
    font-size: 1.9rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}
.verdict-approved { color: #68d391; }
.verdict-rejected { color: #fc8181; }
.result-sub {
    color: #8892a4;
    font-size: 0.9rem;
    font-weight: 300;
    margin-bottom: 1.8rem;
}

/* ── Probability Bar ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    margin-bottom: 0.6rem;
    font-size: 0.82rem;
}
.prob-label { color: #8892a4; width: 70px; text-align: right; font-weight: 600; }
.prob-bar-bg {
    flex: 1;
    height: 8px;
    background: rgba(255,255,255,0.07);
    border-radius: 100px;
    overflow: hidden;
}
.prob-bar-fill-green {
    height: 100%;
    background: linear-gradient(90deg, #38a169, #68d391);
    border-radius: 100px;
    transition: width 0.8s ease;
}
.prob-bar-fill-red {
    height: 100%;
    background: linear-gradient(90deg, #c53030, #fc8181);
    border-radius: 100px;
    transition: width 0.8s ease;
}
.prob-value { width: 42px; font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 0.8rem; }

/* ── Stats Row ── */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.8rem;
    margin-top: 1.4rem;
}
.stat-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 0.9rem;
    text-align: center;
}
.stat-val {
    font-size: 1.2rem;
    font-weight: 700;
    color: #63b3ed;
    font-family: 'JetBrains Mono', monospace;
}
.stat-key { font-size: 0.68rem; color: #8892a4; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.08em; }

/* ── Divider ── */
hr.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 2rem 0;
}

/* ── Risk Pill ── */
.risk-pill {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}
.risk-low    { background: rgba(72,187,120,0.15); color: #68d391; border: 1px solid rgba(72,187,120,0.3); }
.risk-medium { background: rgba(237,137,54,0.15); color: #f6ad55; border: 1px solid rgba(237,137,54,0.3); }
.risk-high   { background: rgba(229,62,62,0.15);  color: #fc8181; border: 1px solid rgba(229,62,62,0.3);  }

/* ── Info note ── */
.info-note {
    background: rgba(99,179,237,0.06);
    border-left: 3px solid #63b3ed;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: #8892a4;
    margin-top: 1rem;
}

/* Columns gap fix */
[data-testid="column"] { padding: 0 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ─── Load Artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = pickle.load(open('best_model.pkl', 'rb'))
    scaler   = pickle.load(open('scaler.pkl', 'rb'))
    encoders = pickle.load(open('encoders.pkl', 'rb'))
    features = pickle.load(open('features.pkl', 'rb'))
    return model, scaler, encoders, features

try:
    model, scaler, encoders, features = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False


# ─── Helper ───────────────────────────────────────────────────────────────────
def encode_input(raw: dict, encoders: dict) -> dict:
    out = raw.copy()
    for col, le in encoders.items():
        if col in out:
            out[col] = le.transform([out[col]])[0]
    return out

def risk_label(prob_reject: float):
    if prob_reject < 0.35:
        return "Low Risk", "risk-low"
    elif prob_reject < 0.65:
        return "Medium Risk", "risk-medium"
    return "High Risk", "risk-high"


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Smart Loan Prediction System</h1>
    <p>Intelligent loan eligibility assessment powered by advanced machine learning — instant, transparent, reliable.</p>
</div>
""", unsafe_allow_html=True)

if not artifacts_loaded:
    st.error("⚠️ Model files not found. Please ensure `best_model.pkl`, `scaler.pkl`, `encoders.pkl`, and `features.pkl` are in the same directory.")
    st.stop()

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ─── Form Layout ──────────────────────────────────────────────────────────────
col_form, col_gap, col_result = st.columns([5, 0.4, 4])

with col_form:
    # ── Personal Info ──
    st.markdown('<div class="section-label">01 / Personal Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Tell us about yourself</div>', unsafe_allow_html=True)

    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            person_age = st.number_input("Age", min_value=18, max_value=100, value=28, step=1)
        with c2:
            person_gender = st.selectbox("Gender", ["male", "female"])

        c3, c4 = st.columns(2)
        with c3:
            person_education = st.selectbox("Education Level",
                ["High School", "Bachelor", "Master", "Doctorate", "Associate"])
        with c4:
            person_home_ownership = st.selectbox("Home Ownership",
                ["RENT", "OWN", "MORTGAGE", "OTHER"])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Financial Info ──
    st.markdown('<div class="section-label">02 / Financial Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Income & Credit Details</div>', unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        person_income = st.number_input("Annual Income ($)", min_value=0, max_value=10_000_000,
                                         value=55000, step=1000)
    with c6:
        person_emp_exp = st.number_input("Work Experience (years)", min_value=0, max_value=60,
                                          value=3, step=1)

    c7, c8 = st.columns(2)
    with c7:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850,
                                        value=650, step=1)
    with c8:
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)",
                                                      min_value=0, max_value=50, value=5, step=1)

    previous_loan_defaults = st.selectbox("Previous Loan Defaults on File", ["No", "Yes"])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Loan Details ──
    st.markdown('<div class="section-label">03 / Loan Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">What are you applying for?</div>', unsafe_allow_html=True)

    c9, c10 = st.columns(2)
    with c9:
        loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=500_000,
                                     value=10000, step=500)
    with c10:
        loan_int_rate = st.slider("Interest Rate (%)", min_value=1.0, max_value=30.0,
                                   value=12.0, step=0.01, format="%.2f")

    loan_intent = st.selectbox("Loan Purpose",
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])

    # Auto-compute loan_percent_income
    loan_percent_income = round(loan_amnt / person_income, 2) if person_income > 0 else 0.0
    st.markdown(f"""
    <div class="info-note">
        💡 <strong>Loan-to-Income Ratio:</strong> {loan_percent_income:.2f}
        &nbsp;({loan_percent_income*100:.1f}% of your annual income)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("  Analyse My Application")


# ─── Prediction ───────────────────────────────────────────────────────────────
with col_result:
    st.markdown('<div class="section-label">04 / Decision</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Eligibility Assessment</div>', unsafe_allow_html=True)

    if predict_btn:
        raw_input = {
            "person_age": float(person_age),
            "person_gender": person_gender,
            "person_education": person_education,
            "person_income": float(person_income),
            "person_emp_exp": int(person_emp_exp),
            "person_home_ownership": person_home_ownership,
            "loan_amnt": float(loan_amnt),
            "loan_intent": loan_intent,
            "loan_int_rate": float(loan_int_rate),
            "loan_percent_income": float(loan_percent_income),
            "cb_person_cred_hist_length": float(cb_person_cred_hist_length),
            "credit_score": int(credit_score),
            "previous_loan_defaults_on_file": previous_loan_defaults,
        }

        encoded = encode_input(raw_input, encoders)
        input_df = pd.DataFrame([encoded])[features]
        scaled   = scaler.transform(input_df)

        prediction = model.predict(scaled)[0]
        probabilities = model.predict_proba(scaled)[0]
        prob_reject  = probabilities[0]
        prob_approve = probabilities[1]

        risk_text, risk_cls = risk_label(prob_reject)

        if prediction == 1:
            st.markdown(f"""
            <div class="result-approved">
                <div class="result-icon">✅</div>
                <div class="result-verdict verdict-approved">Likely Approved</div>
                <div class="result-sub">Your application profile meets the lending criteria.</div>
                <div class="prob-row">
                    <span class="prob-label">Approve</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill-green" style="width:{prob_approve*100:.1f}%"></div>
                    </div>
                    <span class="prob-value" style="color:#68d391">{prob_approve*100:.1f}%</span>
                </div>
                <div class="prob-row">
                    <span class="prob-label">Reject</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill-red" style="width:{prob_reject*100:.1f}%"></div>
                    </div>
                    <span class="prob-value" style="color:#fc8181">{prob_reject*100:.1f}%</span>
                </div>
                <span class="risk-pill {risk_cls}">{risk_text}</span>
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-val">{credit_score}</div>
                        <div class="stat-key">Credit Score</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-val">{loan_percent_income:.2f}</div>
                        <div class="stat-key">Loan / Income</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-val">{loan_int_rate:.1f}%</div>
                        <div class="stat-key">Interest Rate</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-rejected">
                <div class="result-icon">❌</div>
                <div class="result-verdict verdict-rejected">Likely Rejected</div>
                <div class="result-sub">Your profile doesn't meet the current lending criteria.</div>
                <div class="prob-row">
                    <span class="prob-label">Approve</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill-green" style="width:{prob_approve*100:.1f}%"></div>
                    </div>
                    <span class="prob-value" style="color:#68d391">{prob_approve*100:.1f}%</span>
                </div>
                <div class="prob-row">
                    <span class="prob-label">Reject</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill-red" style="width:{prob_reject*100:.1f}%"></div>
                    </div>
                    <span class="prob-value" style="color:#fc8181">{prob_reject*100:.1f}%</span>
                </div>
                <span class="risk-pill {risk_cls}">{risk_text}</span>
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-val">{credit_score}</div>
                        <div class="stat-key">Credit Score</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-val">{loan_percent_income:.2f}</div>
                        <div class="stat-key">Loan / Income</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-val">{loan_int_rate:.1f}%</div>
                        <div class="stat-key">Interest Rate</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Tips ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">💡 Improvement Tips</div>', unsafe_allow_html=True)
        tips = []
        if credit_score < 650:
            tips.append(" Improve your **credit score** above 650 for better approval odds.")
        if loan_percent_income > 0.4:
            tips.append(" Your **loan-to-income ratio** is high — consider a smaller loan amount.")
        if previous_loan_defaults == "Yes":
            tips.append(" **Previous defaults** significantly impact approval. Clear outstanding defaults first.")
        if person_emp_exp < 2:
            tips.append(" Lenders prefer **≥ 2 years** of work experience.")
        if not tips:
            tips.append(" Your profile looks strong! Keep maintaining healthy credit habits.")
        for tip in tips:
            st.markdown(f"<div class='info-note'>{tip}</div>", unsafe_allow_html=True)

    else:
        # Placeholder state
        st.markdown("""
        <div style="
            border: 1.5px dashed rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 3rem 2rem;
            text-align: center;
            margin-top: 0.5rem;
        ">
            <div style="font-size:3rem; margin-bottom:1rem; opacity:0.4;">🔍</div>
            <div style="color:#8892a4; font-size:0.9rem; line-height:1.8;">
                Fill in your details on the left<br>and click <strong style="color:#63b3ed">Analyse My Application</strong><br>to get an instant decision.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:1.5rem;">
            <div class="section-label" style="margin-bottom:0.8rem;">Model Info</div>
        </div>
        """, unsafe_allow_html=True)

        info_items = [
            ("", "Algorithm", "XGBoost Classifier"),
            ("", "Metric", "AUC-optimised"),
            ("", "Features", "13 input variables"),
            ("", "Task", "Binary classification"),
        ]
        for icon, label, val in info_items:
            st.markdown(f"""
            <div style="
                display:flex; align-items:center; gap:0.8rem;
                background:rgba(255,255,255,0.03);
                border:1px solid rgba(255,255,255,0.07);
                border-radius:10px; padding:0.7rem 1rem;
                margin-bottom:0.5rem;
            ">
                <span style="font-size:1.1rem">{icon}</span>
                <span style="color:#8892a4;font-size:0.78rem;width:70px;flex-shrink:0">{label}</span>
                <span style="color:#e8eaf0;font-size:0.85rem;font-weight:600">{val}</span>
            </div>
            """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
