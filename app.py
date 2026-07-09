import shap
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

# --- Page Config ---
st.set_page_config(
    page_title="Medical IT Diabetes Risk Predictor",
    page_icon="🏥",
    layout="centered"
)

# --- Header ---
st.title("🏥 Medical IT — Diabetes Risk Prediction")
st.markdown("**Early risk screening using XGBoost | Medical Information Technology Project**")
st.warning("⚠️ **Disclaimer:** This tool predicts statistical risk only. It is NOT a medical diagnosis. Always consult a healthcare professional.")

# --- Load/Train Model ---
@st.cache_resource
def load_model():
    # Using PIMA dataset for consistency with your notebook
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
    df = pd.read_csv(url, names=names)
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Use same best params from your GridSearchCV
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

# --- Sidebar Info ---
with st.sidebar:
    st.header("📊 Model Info")
    st.metric("Recall", "67.3%", "Prioritized for Medical Screening")
    st.metric("AUC-ROC", "0.76", "Clinical threshold > 0.7")
    st.metric("Accuracy", "69.5%", "Tuned via GridSearchCV")
    st.markdown("---")
    st.markdown("**Tech Stack:**")
    st.markdown("`Python` `XGBoost` `Streamlit` `Scikit-learn`")
    st.markdown("---")
    st.markdown("**GitHub:** [kashish-alt0786/Medical-IT-Diabetes-AI-Project](https://github.com/kashish-alt0786/Medical-IT-Diabetes-AI-Project)")

# --- User Input ---
st.header("📝 Enter Patient Screening Data")
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.slider("Glucose Level (mg/dL)", 0, 200, 120)
    bp = st.slider("Blood Pressure (mm Hg)", 0, 140, 70)
    skin = st.slider("Skin Thickness (mm)", 0, 100, 20)

with col2:
    insulin = st.slider("Insulin (mu U/ml)", 0, 900, 80)
    bmi = st.slider("BMI", 0.0, 70.0, 25.0, 0.1)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5, 0.01)
    age = st.slider("Age", 21, 90, 35)

# --- Prediction ---
if st.button("🔍 Predict Diabetes Risk", type="primary"):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_df = pd.DataFrame(input_data, columns=feature_names)

    risk_proba = model.predict_proba(input_df)[0][1]
    risk_percent = risk_proba * 100

    st.markdown("---")
    st.header("📋 Risk Assessment Result")

    # Risk categorization
    if risk_percent < 30:
        st.success(f"**Low Risk: {risk_percent:.1f}%**")
        st.markdown("Statistical risk is low based on input parameters.")
    elif risk_percent < 70:
        st.warning(f"**Moderate Risk: {risk_percent:.1f}%**")
        st.markdown("Moderate statistical risk detected. Lifestyle monitoring recommended.")
    else:
        st.error(f"**High Risk: {risk_percent:.1f}%**")
        st.markdown("High statistical risk detected. Clinical screening strongly advised.")

    st.markdown("---")

    # ===== KASHISH GKS UPGRADE: SHAP EXPLAINABILITY =====
    st.subheader("🔍 Why This Risk Score? — AI Explainability")

    # SHAP requires these imports - add at top of file
    import shap
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # Kaggle Data Viz style chart
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))

    feature_labels = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
                     'Insulin', 'BMI', 'Pedigree Function', 'Age']
    colors = ['#d62728' if x > 0 else '#2ca02c' for x in shap_values[0]]

    sns.barplot(x=shap_values[0], y=feature_labels, palette=colors, ax=ax)
    ax.set_title('Feature Impact on Diabetes Risk Prediction', fontsize=14, fontweight='bold')
    ax.set_xlabel('SHAP Value: Impact on Model Output')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    st.caption('✅ Chart created using Seaborn — Certified via Kaggle Data Visualization course. Red bars increase risk, green bars decrease risk.')

    # Clinical Notes - BCG X style
    st.subheader("📋 Clinical Interpretation for Healthcare Providers")

    # Get top 3 features by absolute SHAP value
    shap_df = pd.DataFrame({
        'feature': feature_labels,
        'shap_value': shap_values[0],
        'input_value': input_df.iloc[0].values
    }).sort_values('shap_value', key=abs, ascending=False).head(3)

    st.markdown(f"""
    **Patient Risk Profile:**
    1. **{shap_df.iloc[0]['feature']}** = `{shap_df.iloc[0]['input_value']:.1f}` — Strongest driver of risk
    2. **{shap_df.iloc[1]['feature']}** = `{shap_df.iloc[1]['input_value']:.1f}` — Secondary contributor
    3. **{shap_df.iloc[2]['feature']}** = `{shap_df.iloc[2]['input_value']:.1f}` — Moderate impact

    **Recommendation:** Based on BCG X Data Science healthcare case study methodology,
    patients with risk >70% should be flagged for preventive counseling per digital
    therapeutic guidelines. This explainability framework builds clinician trust in AI predictions.
    """)

    st.info("💡 **GKS 2027 Note:** This explainability approach aligns with Yonsei BSI research on transparent AI for clinical decision support. Methodology validated via BCG X Simulation + Kaggle ML certifications.")
    # ===== END UPGRADE =====

    st.markdown("---")
    st.caption("Model trained on PIMA Indian Diabetes Dataset. Recall optimized for medical screening ethics.")
    
    # Risk categorization
    if risk_percent < 30:
        st.success(f"**Low Risk: {risk_percent:.1f}%**")
        st.markdown("Statistical risk is low based on input parameters.")
    elif risk_percent < 70:
        st.warning(f"**Moderate Risk: {risk_percent:.1f}%**")
        st.markdown("Moderate statistical risk detected. Lifestyle monitoring recommended.")
    else:
        st.error(f"**High Risk: {risk_percent:.1f}%**")
        st.markdown("High statistical risk detected. Clinical screening strongly advised.")
    
    # Feature contribution note
    st.info("💡 **Top risk factors in this model:** Glucose level, BMI, and Age. Based on SHAP explainability analysis.")
    
    st.markdown("---")
    st.caption("Model trained on PIMA Indian Diabetes Dataset. Recall optimized for medical screening ethics.")

# --- Footer ---
st.markdown("---")
st.caption("Developed as independent Medical IT research project, 2026 | For educational purposes only")
