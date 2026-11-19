import joblib
import pandas as pd
import streamlit as st
from src.features import add_features

st.set_page_config(page_title="Loan Eligibility Predictor", page_icon="🏦")
st.title("Loan Eligibility Predictor")
st.caption("Enter applicant details to estimate approval probability.")


@st.cache_resource
def load_model():
    return joblib.load("models/loan_pipeline.joblib")


pipeline = load_model()

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self employed", ["No", "Yes"])
    property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
with col2:
    app_income = st.number_input("Applicant monthly income", min_value=0, value=5000, step=500)
    coapp_income = st.number_input("Coapplicant monthly income", min_value=0, value=0, step=500)
    loan_amount = st.number_input("Loan amount (in thousands)", min_value=1, value=130)
    term = st.selectbox("Loan term (months)", [360, 180, 480, 300, 240, 120, 84, 60, 36, 12])
    credit = st.radio("Credit history meets guidelines?", ["Yes", "No"], horizontal=True)

if st.button("Check eligibility"):
    row = pd.DataFrame([{
        "Gender": gender, "Married": married, "Dependents": dependents,
        "Education": education, "Self_Employed": self_employed,
        "ApplicantIncome": app_income, "CoapplicantIncome": coapp_income,
        "LoanAmount": loan_amount, "Loan_Amount_Term": term,
        "Credit_History": 1.0 if credit == "Yes" else 0.0,
        "Property_Area": property_area,
    }])
    row = add_features(row)
    prob = pipeline.predict_proba(row)[0, 1]

    st.progress(float(prob))
    st.metric("Approval probability", f"{prob:.1%}")
    if prob >= 0.5:
        st.success("Likely to be approved")
    else:
        st.error("Higher credit risk, likely to be rejected")
        
st.divider()
# st.caption("(👉ﾟヮﾟ)👉 Work Done by Muzammil Maqsood aka Astra Zombater. ( ͡~ ͜ʖ ͡°)")
st.divider()

st.markdown(
    """
    <div style="text-align: center; color: green; font-size: 0.875rem; line-height: 1.8;">
        (👉ﾟヮﾟ)👉 Work Done by Muzammil Maqsood aka Astra Zombater. ( ͡~ ͜ʖ ͡°) <br>
        <div style="display: inline-flex; align-items: center; gap: 15px; margin-top: 4px;">
            <a href="mailto:muzammilmaqsood522@gmail.com" style="color: green; text-decoration: none; display: inline-flex; align-items: center; gap: 5px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect width="20" height="16" x="2" y="4" rx="2"></rect>
                    <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"></path>
                </svg>
                Email
            </a>
            <span>|</span>
            <a href="https://github.com/astra-zombater" target="_blank" style="color: green; text-decoration: none; display: inline-flex; align-items: center; gap: 5px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
                </svg>
                GitHub
            </a>
            <span>|</span>
            <a href="https://www.instagram.com/jrnomiboltaa/" target="_blank" style="color: green; text-decoration: none; display: inline-flex; align-items: center; gap: 5px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect width="20" height="20" x="2" y="2" rx="5" ry="5"></rect>
                    <path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z"></path>
                    <line x1="17.5" x2="17.51" y1="6.5" y2="6.5"></line>
                </svg>
                Instagram
            </a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
