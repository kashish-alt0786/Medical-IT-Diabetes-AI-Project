import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Health Risk Checker", page_icon="❤️", layout="centered")

st.title("❤️ Health Risk Checker")
st.caption("Check if you might be at risk for blood sugar problems")
st.warning("⚠️ This is not a doctor. Always ask an adult or doctor if you're worried.")

# --- Load Model ---
@st.cache_resource
def load_model():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
    df = pd.read_csv(url, names=names)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()

model, feature_names = load_model()

st.divider()
st.subheader("📝 Answer these easy questions")

# --- SUPER SIMPLE INPUTS ---
age = st.slider("1. How old are you?", 5, 100, 12)

st.markdown("**2. Have you done a blood sugar test?**")
knows_glucose = st.radio("Blood sugar test", ["No", "Yes"], horizontal=True, label_visibility="collapsed")

if knows_glucose == "Yes":
    glucose = st.number_input("Type your blood sugar number here", 50, 300, 90)
    st.caption("Normal is usually 70-100")
else:
    st.markdown("**No test? Answer these:**")
    thirsty = st.checkbox("I feel very thirsty a lot")
    tired = st.checkbox("I feel very tired even after sleeping")
    pee = st.checkbox("I go to the bathroom to pee a lot")
    # Simple symptom scoring
    symptom_score = sum([thirsty, tired, pee])
    glucose = 85 + symptom_score * 15 # 85, 100, 115, 130
    st.info(f"We'll use {glucose} as an estimate based on your answers")

st.markdown("**3. Your body size**")
col1, col2 = st.columns(2)
height = col1.number_input("How tall are you? (cm)", 100, 220, 150)
weight = col2.number_input("How much do you weigh? (kg)", 20, 150, 45)
bmi = weight / ((height/100)**2)

st.markdown("**4. Your family**")
family = st.radio("Does your mom, dad, brother, or sister have diabetes?", 
                  ["No", "Yes", "I don't know"])
dpf = 0.15 if family == "No" else 1.2 if family == "Yes" else 0.3

st.markdown("**5. For girls only**")
gender = st.radio("Are you a boy or girl?", ["Boy", "Girl"])
if gender == "Girl":
    pregnancies = st.number_input("How many times have you been pregnant?", 0, 10, 0)
else:
    pregnancies = 0

# Hidden defaults
bp = 80
insulin = 80
skin = 20

st.divider()

# --- Prediction ---
if st.button("🔍 Check My Risk", type="primary", use_container_width=True):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_df = pd.DataFrame(input_data, columns=feature_names)

    risk_proba = model.predict_proba(input_df)[0][1]
    risk_percent = int(risk_proba * 100)

    st.markdown("---")
    st.header("📋 Your Result")

    if risk_percent < 30:
        st.success(f"**🟢 Low Risk: {risk_percent}%**")
        st.markdown("**This means:** Your chance is low. Keep eating healthy and playing!")
        st.balloons()
    elif risk_percent < 70:
        st.warning(f"**🟡 Medium Risk: {risk_percent}%**")
        st.markdown("**This means:** You should tell an adult. Eating less sugar and exercising helps.")
    else:
        st.error(f"**🔴 High Risk: {risk_percent}%**")
        st.markdown("**This means:** Please tell your parents and see a doctor to check.")

    # Simple explanation instead of SHAP
    st.markdown("---")
    st.subheader("🤔 Why did I get this result?")
    
    reasons = []
    if glucose > 120: reasons.append("• Your blood sugar number was
