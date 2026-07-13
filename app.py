import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Config ---
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# --- Header ---
st.title("🩺 Diabetes Risk Predictor")
st.caption("Explainable AI for preventive health screening")
st.warning("⚠ **Disclaimer:** This tool predicts statistical risk only. It is NOT a medical diagnosis. Always consult a healthcare professional.")

# --- Load/Train Model ---
@st.cache_resource
def load_model():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
    df = pd.read_csv(url, names=names)
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    scale_pos_weight = len(y[y==0]) / len(y[y==1])
    model = xgb.XGBClassifier(
        learning_rate=0.1,
        max_depth=3,
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X, y)
    return model, X.columns.tolist()

model, feature_names = load_model()

# --- Sidebar ---
with st.sidebar:
    st.header("📊 Model Performance")
    st.metric("Recall", "67.3%", "Optimized for Medical Screening")
    st.metric("AUC-ROC", "0.76", "Clinical threshold > 0.7")
    st.metric("Accuracy", "69.5%", "Tuned via GridSearchCV")
    st.markdown("---")
    st.markdown("**Project Links:**")
    st.markdown("[📓 Kaggle Research](https://www.kaggle.com/code/kashish0000000/notebookb6b8ef2c97)")
    st.markdown("[💻 GitHub Code](https://github.com/kashish-alt0786/Medical-IT-Diabetes-AI-Project)")
    st.markdown("---")
    st.markdown("**Tech Stack:** `Python` `XGBoost` `SHAP` `Streamlit`")

# --- User Input ---
st.header("📝 Health Information")

col1, col2 = st.columns(2)
age = col1.number_input("Age", 1, 120, 30, help="Your current age")

# ===== ONLY THIS PART
