import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load model and features
model = joblib.load("hospital_readmission_model.pkl")
model_features = joblib.load("model_features.pkl")

# Title
st.title("🏥 Hospital Readmission Risk Predictor")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file with patient data", type=["csv"])

# Preprocessing function (update as per your training code)
def preprocess_user_input(df):
    df = df.copy()
    
    # Drop unnecessary columns if they exist
    drop_cols = ["encounter_id", "patient_nbr"]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')
    
    # Handle missing values
    df.fillna("missing", inplace=True)
    
    # One-hot encoding for categorical features
    df = pd.get_dummies(df)
    
    return df

if uploaded_file is not None:
    # Load the CSV
    input_df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview:")
    st.dataframe(input_df.head())

    # Preprocess
    preprocessed_df = preprocess_user_input(input_df)

    # Align columns with training features
    input_aligned = preprocessed_df.reindex(columns=model_features, fill_value=0)

    # Prediction
    prediction = model.predict(input_aligned)

    # Show results
    st.subheader("Predictions:")
    result_df = input_df.copy()
    result_df["Readmission_Prediction"] = prediction
    st.dataframe(result_df[["readmitted", "Readmission_Prediction"]] if "readmitted" in result_df.columns else result_df)

    # Analysis
    st.subheader("Prediction Summary:")
    st.write(result_df["Readmission_Prediction"].value_counts())
else:
    st.info("Please upload a CSV file to begin.")
