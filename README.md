# 🏥 Hospital Readmission Risk Predictor

    An end-to-end ""machine learning + Streamlit"" application that predicts ""30-day hospital readmission risk"" using patient encounter data and presents results through an interactive dashboard.

    This project demonstrates ""real-world ML deployment"", feature engineering, and decision-support thinking for healthcare analytics.

## 🚀 Live Demo

    👉 https://hospital-readmission-predictor-kro4jvk9njwzt3a4xitsbj.streamlit.app/

## 📌 Key Features

    - Predicts probability of 30-day hospital readmission
    - Interactive Streamlit dashboard with real-time inputs
    - Trained ""Random Forest Classifier"" with class imbalance handling
    - Clear risk categorization (High Risk vs Low Risk)
    - Estimated cost savings insight for high-risk prevention
    - Production-ready project structure

## 🧠 Machine Learning Overview

    - Model: RandomForestClassifier
    - Class Weighting: Balanced
    - Evaluation Focus: Recall & probability-based decision threshold
    - Target: Hospital readmission within 30 days

    Model Features Used:

    - `age_numeric`
    - `time_in_hospital`
    - `num_lab_procedures`
    - `num_medications`
    - `number_emergency`
    - `number_inpatient`

## 🖥️ Tech Stack

    - Python
    - Pandas, NumPy
    - Scikit-learn
    - Streamlit
    - Joblib
    - Jupyter Notebook

## 📁 Project Structure

    Hospital-Readmission-Predictor/
    │
    ├── app/
    │   └── app.py              # Streamlit application
    │
    ├── models/
    │   └── rf_readmission_model.pkl
    │
    ├── notebooks/
    │   └── 01_data_exploration.ipynb
    │
    ├── data/                   # Sample / reference data
    │
    ├── requirements.txt
    ├── .gitignore
    └── README.md
    
⚙️ How to Run Locally

    # 1. Clone the repository
    git clone https://github.com/YashJadhav100/Hospital-Readmission-Predictor.git

    # 2. Navigate to project
    cd Hospital-Readmission-Predictor

    # 3. Install dependencies
    pip install -r requirements.txt

    # 4. Run Streamlit app
    streamlit run app/app.py

📊 Use Case

    Healthcare providers can use this tool to:

    Identify high-risk patients early

    Optimize discharge planning

    Reduce avoidable readmissions

    Improve patient outcomes

    Estimate potential cost savings

👤 Author

    Yash Jadhav
    Graduate Student – Computer Science
    Syracuse University

    🔗 GitHub: https://github.com/YashJadhav100

    🔗 LinkedIn: https://www.linkedin.com/in/yashvjadhav/

📄 Disclaimer

    This project is for educational and analytical purposes only and does not constitute medical advice.




