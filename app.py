import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Config ---
st.set_page_config(
    page_title="Medical IT Diabetes Risk Predictor | GKS 2027",
    page_icon="🏥",
    layout="centered"
)

# --- Header — GKS BRANDING ---
st.title("🏥 Medical IT — Diabetes Risk Predictor")
st.markdown("**Built by Kashish | Explainable AI for Preventive Healthcare**")
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

# --- Sidebar — GKS PORTFOLIO ---
with st.sidebar:
    st.header("📊 Model Performance")
    st.metric("Recall", "67.3%", "Optimized for Medical Screening")
    st.metric("AUC-ROC", "0.76", "Clinical threshold > 0.7")
    st.metric("Accuracy", "69.5%", "Tuned via GridSearchCV")
    st.markdown("---")
    st.markdown("**GKS Links:**")
    st.markdown("[📓 Kaggle Research](https://www.kaggle.com/code/kashish0000000/notebookb6b8ef2c97)")
    st.markdown("[💻 GitHub Code](https://github.com/kashish-alt0786/Medical-IT-Diabetes-AI-Project)")
    st.markdown("---")
    st.markdown("**Tech Stack:** `Python` `XGBoost` `SHAP` `Streamlit`")

# --- User Input — ACE VERSION: NO CONFUSING FIELDS ---
st.header("📝 Step 1: Basic Info — Everyone Knows This")
col1, col2 = st.columns(2)
age = col1.number_input("Age", 1, 120, 30)
glucose = col2.number_input("Fasting Glucose (mg/dL)", 50, 300, 120, 
    help="Normal is 70-100. If you ate in last 2 hrs, add 30 to your reading")

st.subheader("📏 Step 2: Body Info — We Calculate BMI For You")
col3, col4 = st.columns(2)
height = col3.number_input("Height (cm)", 100, 250, 165)
weight = col4.number_input("Weight (kg)", 30, 200, 65)
bmi = weight / ((height/100)**2)
st.success(f"Your BMI: {bmi:.1f} | {'Normal' if bmi<25 else 'Overweight' if bmi<30 else 'Obese'}")

st.subheader("❤️ Step 3: Health Background")
col5, col6 = st.columns(2)
bp_option = col5.selectbox("Blood Pressure", 
    ["Normal / No issues", "High / I take BP medicine", "Don't Know"])
bp = 80 if "Normal" in bp_option else 100 if "High" in bp_option else 85

pregnancies = col6.number_input("Number of Pregnancies", 0, 20, 0, 
    help="Enter 0 if male or not applicable")

pedigree = st.radio("Do parents or siblings have diabetes?", 
    ["No", "Yes", "Not Sure"], horizontal=True)
dpf = 0.8 if pedigree == "Yes" else 0.3

# Clinical safe defaults for fields users never know
insulin = 80 # Median from PIMA dataset
skin = 20 # Median from PIMA dataset

st.divider()

# --- Prediction ---
if st.button("🔍 Predict My Diabetes Risk", type="primary", use_container_width=True):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_df = pd.DataFrame(input_data, columns=feature_names)

    risk_proba = model.predict_proba(input_df)[0][1]
    risk_percent = risk_proba * 100

    st.markdown("---")
    st.header("📋 Risk Assessment Result")

    # Risk categorization — ONLY ONCE
    if risk_percent < 30:
        st.success(f"**Low Risk: {risk_percent:.1f}%**")
        st.markdown("Statistical risk is low. Maintain healthy lifestyle and recheck annually.")
    elif risk_percent < 70:
        st.warning(f"**Moderate Risk: {risk_percent:.1f}%**")
        st.markdown("Moderate risk detected. Consider lifestyle changes and glucose monitoring.")
    else:
        st.error(f"**High Risk: {risk_percent:.1f}%**")
        st.markdown("High risk detected. Clinical screening with HbA1c test strongly advised.")

    st.markdown("---")

    # SHAP Explainability — GKS UPGRADE
    st.subheader("🔍 Why This Risk Score? — AI Explainability")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

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
    plt.close()
    
    st.caption('✅ Red bars increase risk, green bars decrease risk. Based on SHAP explainable AI.')

    # Clinical Interpretation
    st.subheader("📋 Clinical Interpretation")
    shap_df = pd.DataFrame({
        'feature': feature_labels,
        'shap_value': shap_values[0],
        'input_value': input_df.iloc[0].values
    }).sort_values('shap_value', key=abs, ascending=False).head(3)

    st.markdown(f"""
    **Top 3 Risk Drivers:**
    1. **{shap_df.iloc[0]['feature']}** = `{shap_df.iloc[0]['input_value']:.1f}` — Strongest impact
    2. **{shap_df.iloc[1]['feature']}** = `{shap_df.iloc[1]['input_value']:.1f}` — Secondary factor 
    3. **{shap_df.iloc[2]['feature']}** = `{shap_df.iloc[2]['input_value']:.1f}` — Moderate impact
    
    **Medical IT Note:** This explainability aligns with Korea’s Digital Healthcare Innovation Strategy for transparent AI in clinical decision support.
    """)

    # Link to Project 2 — GKS STORY
    st.divider()
    st.subheader("🎯 Next Step: From Detection to Prevention")
    st.markdown("**High risk?** Your daily meals impact blood sugar more than genetics.")
    st.link_button("Open NutriGuard AI — Check If Your Meals Are Safe →", 
        "https://github.com/kashish-alt0786") # Update after NutriGuard is live
    st.caption("Coming Soon: AI nutritionist for Korean + Indian food |

# --- Footer ---
st.divider()
st.caption("Disclaimer: For educational and informational purposes only. Not medical advice. Model trained on Pima Indian Diabetes Dataset.")
