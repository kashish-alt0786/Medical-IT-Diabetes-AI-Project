"""Explainable AI and what-if analysis page."""

import numpy as np
import pandas as pd
import streamlit as st

from model_training import FEATURE_NAMES, TrainingResult, train_and_compare

st.set_page_config(page_title="XAI Diagnostic Room", page_icon="🔬", layout="wide")

st.title("🔬 XAI Diagnostic Room")
st.caption("Explore how the selected machine-learning model responds to individual health inputs.")

st.warning(
    "Educational screening tool only. This page does not diagnose diabetes or replace professional medical care."
)

@st.cache_resource(show_spinner=False)
def get_training_result() -> TrainingResult:
    return train_and_compare()

with st.spinner("Loading the validated model and explanation engine..."):
    result = get_training_result()

st.success(f"Selected model: **{result.best_name}** | Selection priority: F1-score → Recall → ROC-AUC")

st.subheader("1. Patient profile")
cols = st.columns(4)
with cols[0]:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.slider("Glucose (mg/dL)", 50, 250, 120)
with cols[1]:
    bp = st.slider("Blood Pressure (mm Hg)", 40, 140, 70)
    skin = st.slider("Skin Thickness (mm)", 5, 80, 25)
with cols[2]:
    insulin = st.slider("Insulin (µU/mL)", 10, 700, 100)
    bmi = st.slider("BMI", 15.0, 60.0, 30.0, 0.1)
with cols[3]:
    dpf = st.slider("Diabetes Pedigree Function", 0.05, 2.5, 0.45, 0.01)
    age = st.slider("Age", 21, 100, 30)

input_df = pd.DataFrame([{
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": bp,
    "SkinThickness": skin,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": dpf,
    "Age": age,
}])

model = result.best_model
probability = float(model.predict_proba(input_df)[0, 1])
risk = probability * 100
predicted_class = int(probability >= result.best_threshold)

m1, m2, m3 = st.columns(3)
m1.metric("Calculated Risk", f"{risk:.1f}%")
m2.metric("Screening Class", "Higher-risk" if predicted_class else "Lower-risk")
m3.metric("Decision Threshold", f"{result.best_threshold:.2f}")

st.subheader("2. SHAP explanation")
try:
    import shap
    import matplotlib.pyplot as plt

    preprocess = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    transformed = preprocess.transform(input_df)
    transformed_names = list(preprocess.get_feature_names_out())

    if hasattr(estimator, "get_booster"):
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    elif hasattr(estimator, "feature_importances_"):
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        explainer = shap.LinearExplainer(estimator, transformed)
        shap_values = explainer.shap_values(transformed)

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    values = shap_values[0]
    order = np.argsort(np.abs(values))[::-1]
    top = order[:8]
    display_names = [name.replace("numeric__", "") for name in np.array(transformed_names)[top]]
    display_values = values[top]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(display_names[::-1], display_values[::-1])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("SHAP contribution to model output")
    ax.set_title("Individual Feature Contributions")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.caption("Positive values push the model toward higher predicted risk; negative values push it toward lower predicted risk.")
except Exception as exc:
    st.info(f"The selected model could not produce a SHAP plot in this runtime: {exc}")

st.subheader("3. What-if analysis")
st.write("Change one or more inputs below and see how the model's risk estimate changes.")

what_if = input_df.copy()
what_if["Glucose"] = st.slider("What-if glucose", 50, 250, int(glucose), key="whatif_glucose")
what_if["BMI"] = st.slider("What-if BMI", 15.0, 60.0, float(bmi), 0.1, key="whatif_bmi")
what_if["Age"] = st.slider("What-if age", 21, 100, int(age), key="whatif_age")

what_if_probability = float(model.predict_proba(what_if)[0, 1])
change = (what_if_probability - probability) * 100
wc1, wc2 = st.columns(2)
wc1.metric("What-if Risk", f"{what_if_probability * 100:.1f}%")
wc2.metric("Change from baseline", f"{change:+.1f} percentage points")

st.info("What-if results are model outputs, not predictions of how a real person's health will change. Clinical decisions require qualified medical advice.")
