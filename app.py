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

# --- User Input — 100% USER-FRIENDLY ---
st.header("📝 Health Information")

col1, col2 = st.columns(2)
age = col1.number_input("Age", 1, 120, 30, help="Your current age")
glucose = col2.number_input("Fasting Glucose (mg/dL)", 50, 300, 100, 
    help="Normal range: 70-100. If tested after a meal, add 30 to your reading")

st.subheader("📏 Body Measurements")
col3, col4 = st.columns(2)
height = col3.number_input("Height (cm)", 100, 250, 165)
weight = col4.number_input("Weight (kg)", 30, 200, 65)
bmi = weight / ((height/100)**2)
st.info(f"Calculated BMI: {bmi:.1f} | {'Normal' if bmi<25 else 'Overweight' if bmi<30 else 'Obese'}")

st.subheader("❤️ Health Background")
col5, col6 = st.columns(2)
bp_option = col5.selectbox("Blood Pressure Status", 
    ["Normal", "High Blood Pressure", "Not Sure"])
bp = 80 if bp_option == "Normal" else 100 if bp_option == "High Blood Pressure" else 85

pregnancies = col6.number_input("Number of Pregnancies", 0, 20, 0, 
    help="Enter 0 if male or not applicable")

family_history = st.radio(
    "Do any parents, siblings, or children have diabetes?",
    ["No", "Yes, 1 family member", "Yes, 2 or more family members", "Not Sure"],
    horizontal=True,
    help="This helps assess genetic risk"
)

# Convert family history to DPF value
if family_history == "No":
    dpf = 0.15
elif family_history == "Yes, 1 family member":
    dpf = 0.5
elif family_history == "Yes, 2 or more family members":
    dpf = 1.2
else:
    dpf = 0.3

# Clinical safe defaults for fields users typically don't know
insulin = 80
skin = 20

st.divider()

# --- Prediction ---
if st.button("🔍 Analyze My Risk", type="primary", use_container_width=True):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_df = pd.DataFrame(input_data, columns=feature_names)

    risk_proba = model.predict_proba(input_df)[0][1]
    risk_percent = risk_proba * 100

    st.markdown("---")
    st.header("📋 Risk Assessment Result")

    if risk_percent < 30:
        st.success(f"**Lower Risk: {risk_percent:.1f}%**")
        st.markdown("Your statistical risk is low. Maintaining a healthy lifestyle is recommended.")
    elif risk_percent < 70:
        st.warning(f"**Moderate Risk: {risk_percent:.1f}%**")
        st.markdown("Your statistical risk is moderate. Consider lifestyle monitoring and regular health checkups.")
    else:
        st.error(f"**Elevated Risk: {risk_percent:.1f}%**")
        st.markdown("Your statistical risk is elevated. Consulting a healthcare professional for further testing is strongly advised.")

    st.markdown("---")

    # SHAP Explainability
    st.subheader("🔬 How This Result Was Calculated")
    st.caption("The chart shows which factors increased or decreased your risk score:")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))
    feature_labels = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
                     'Insulin', 'BMI', 'Family History', 'Age']
    colors = ['#d62728' if x > 0 else '#2ca02c' for x in shap_values[0]]

    sns.barplot(x=shap_values[0], y=feature_labels, palette=colors, ax=ax)
    ax.set_title('Feature Impact on Risk Prediction', fontsize=14, fontweight='bold')
    ax.set_xlabel('Impact on Model Output')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()
    
    st.caption('Red bars increase risk. Green bars decrease risk.')

    # Clinical Interpretation
    st.subheader("📋 Key Risk Factors")
    shap_df = pd.DataFrame({
        'feature': feature_labels,
        'shap_value': shap_values[0],
        'input_value': input_df.iloc[0].values
    }).sort_values('shap_value', key=abs, ascending=False).head(3)

    st.markdown(f"""
    **Top 3 factors influencing your result:**
    1. **{shap_df.iloc[0]['feature']}**: `{shap_df.iloc[0]['input_value']:.1f}`
    2. **{shap_df.iloc[1]['feature']}**: `{shap_df.iloc[1]['input_value']:.1f}`
    3. **{shap_df.iloc[2]['feature']}**: `{shap_df.iloc[2]['input_value']:.1f}`
    
    *This explainability helps users and healthcare providers understand the prediction.*
    """)

# --- Footer ---
st.divider()
st.caption("Disclaimer: For educational and informational purposes only. Not medical advice. Model trained on Pima Indian Diabetes Dataset.")
st.caption("Built with Python, Streamlit, XGBoost, SHAP.")
