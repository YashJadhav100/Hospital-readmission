import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Hospital Readmission Risk Predictor",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Hospital Readmission Risk Predictor")
st.write(
    "Predict 30-day hospital readmission risk using a trained Random Forest model."
)

# -----------------------------
# Load model (safe path)
# -----------------------------
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "models" / "rf_readmission_model.pkl"
    return joblib.load(model_path)

model = load_model()

# -----------------------------
# Sidebar inputs (MATCH MODEL)
# -----------------------------
st.subheader("🧑‍⚕️ Patient Information")

age_numeric = st.slider("Age", 18, 100, 60)
time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 5)
num_lab_procedures = st.slider("Number of Lab Procedures", 0, 100, 40)
num_medications = st.slider("Number of Medications", 1, 50, 10)
number_inpatient = st.slider("Number of Inpatient Visits", 0, 10, 1)
number_emergency = st.slider("Number of Emergency Visits", 0, 10, 2)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Readmission Risk"):
    # Build input EXACTLY as model expects
    input_data = pd.DataFrame([{
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_medications": num_medications,
        "number_inpatient": number_inpatient,
        "number_emergency": number_emergency,
        "age_numeric": age_numeric
    }])

    proba = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.metric("Readmission Risk Probability", f"{proba:.2%}")

    if proba >= 0.35:
        st.error("⚠️ High Risk Patient")
        st.write("💰 Estimated cost savings if prevented: **$15,000**")
    else:
        st.success("✅ Low Risk Patient")
