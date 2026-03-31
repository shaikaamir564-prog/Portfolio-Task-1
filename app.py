import streamlit as st
import numpy as np
import pickle

# -----------------------------
# LOAD FILES
# -----------------------------
model = pickle.load(open('best_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
features = pickle.load(open('features.pkl', 'rb'))

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Loan Approval System",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS (Professional UI)
# -----------------------------
st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
    }
    .main-title {
        font-size: 32px;
        font-weight: bold;
        color: #2c3e50;
    }
    .section-title {
        font-size: 20px;
        margin-top: 20px;
        color: #34495e;
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    '<div class="main-title"><i class="fa fa-bank"></i> Loan Approval Prediction System</div>',
    unsafe_allow_html=True
)

st.write("Predict whether a loan application should be approved based on applicant information.")

# -----------------------------
# INPUT SECTION
# -----------------------------
st.markdown('<div class="section-title"><i class="fa fa-user"></i> Applicant Details</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education", ["High School", "Associate", "Bachelor", "Master"])

with col2:
    income = st.number_input("Annual Income", value=50000)
    experience = st.number_input("Employment Experience (years)", value=2)
    home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])

with col3:
    loan_amount = st.number_input("Loan Amount", value=10000)
    intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT"])
    interest_rate = st.number_input("Interest Rate (%)", value=10.0)

# -----------------------------
# SECOND ROW INPUTS
# -----------------------------
col4, col5, col6 = st.columns(3)

with col4:
    percent_income = st.number_input("Loan Percent Income", value=0.2)

with col5:
    credit_history = st.number_input("Credit History Length", value=3)

with col6:
    credit_score = st.number_input("Credit Score", value=650)
    previous_default = st.selectbox("Previous Default", ["No", "Yes"])

# -----------------------------
# ENCODING (MATCH TRAINING)
# -----------------------------
gender = 1 if gender == "Male" else 0

education_map = {"High School":0, "Associate":1, "Bachelor":2, "Master":3}
education = education_map[education]

home_map = {"OWN":0, "MORTGAGE":1, "RENT":2}
home = home_map[home]

intent_map = {
    "PERSONAL":0,
    "EDUCATION":1,
    "MEDICAL":2,
    "VENTURE":3,
    "HOMEIMPROVEMENT":4
}
intent = intent_map[intent]

previous_default = 1 if previous_default == "Yes" else 0

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
st.markdown('<div class="section-title"><i class="fa fa-cogs"></i> Prediction</div>', unsafe_allow_html=True)

if st.button("Predict Loan Status"):

    input_data = np.array([[age, gender, education, income, experience,
                            home, loan_amount, intent, interest_rate,
                            percent_income, credit_history, credit_score,
                            previous_default]])

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # -----------------------------
    # OUTPUT CARD
    # -----------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(
            f"<h3 style='color:green;'><i class='fa fa-check-circle'></i> Loan Approved</h3>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<h3 style='color:red;'><i class='fa fa-times-circle'></i> Loan Rejected</h3>",
            unsafe_allow_html=True
        )

    st.write(f"**Prediction Confidence:** {probability:.2f}")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.write("Developed as part of Machine Learning System Project")
