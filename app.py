import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Risk Prediction System")
st.markdown("Predict **default probability**, **credit score**, and **risk segment**")

st.divider()

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_artifacts():
    with open("gb_model (2).pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler (1).pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

# ------------------ FUNCTIONS ------------------
def create_scorecard(prob):
    score = 850 - (prob * 550)
    return int(np.clip(score, 300, 850))

# ------------------ INPUT UI ------------------
with st.form("credit_form"):
    st.subheader("📝 Applicant Details")

    limit_bal = st.number_input(
        "Limit Balance",
        min_value=0.0,
        step=1000.0
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100
    )

    sex = st.selectbox(
        "Sex",
        options={1: "Male", 2: "Female"},
        format_func=lambda x: f"{x} - {'Male' if x == 1 else 'Female'}"
    )

    education = st.selectbox(
        "Education",
        options={
            1: "Graduate School",
            2: "University",
            3: "High School",
            4: "Others",
            5: "Unknown",
            6: "Unknown"
        }
    )

    marriage = st.selectbox(
        "Marital Status",
        options={
            1: "Married",
            2: "Single",
            3: "Others"
        }
    )

    submitted = st.form_submit_button("🔍 Predict Credit Risk")

# ------------------ PREDICTION ------------------
if submitted:
    try:
        input_data = pd.DataFrame([[
            limit_bal, sex, education, marriage, age,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0
        ]], columns=[
            'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
            'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
            'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
            'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
            'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
        ])

        # Feature engineered column
        input_data["TOTAL_BILL_AMT"] = input_data[
            ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
        ].sum(axis=1)

        scaled_input = scaler.transform(input_data)

        prob_default = model.predict_proba(scaled_input)[0][1]
        credit_score = create_scorecard(prob_default)

        if credit_score >= 700:
            risk = "🟢 Low Risk"
        elif credit_score >= 600:
            risk = "🟡 Medium Risk"
        else:
            risk = "🔴 High Risk"

        st.divider()
        st.subheader("📊 Prediction Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Default Probability", f"{prob_default:.4f}")
        col2.metric("Credit Score", credit_score)
        col3.metric("Risk Segment", risk)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
