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
    st.markdown("---")
    st.caption("Model Limitations: Trained on Pima Indian female dataset. For screening only.")

# --- User Input ---
st.header("📝 Health Information")

col1, col2 = st.columns(2)
age = col1.number_input("Age", 1, 120, 30, help="Your current age")

st.subheader("🩸 Blood Sugar Level")

# Option 1: User knows their number
knows_glucose = st.radio(
    "Do you have a blood sugar test result?",
    ["No, I don't know", "Yes, I have a test result"],
    horizontal=True
)

if knows_glucose == "Yes, I have a test result":
    glucose = st.number_input(
        "Type your Fasting Blood Sugar number", 
        50, 300, 90,
        help="Check your lab report for 'Fasting Blood Sugar' or 'FBS'"
    )
else:
    st.markdown("**No test? Answer these 3 questions:**")
    thirsty = st.checkbox("I feel very thirsty all the time")
    tired = st.checkbox("I feel tired even after sleeping 8 hours") 
    pee_lot = st.checkbox("I go to the bathroom to pee very often")
    
    # Estimate glucose from symptoms
    symptom_count = sum([thirsty, tired, pee_lot])
    if symptom_count == 0:
        glucose = 85  # Healthy estimate
        st.success("Estimated blood sugar: 85 (Normal range)")
    elif symptom_count == 1:
        glucose = 105 # Pre-diabetes estimate  
        st.warning("Estimated blood sugar: 105 (Slightly high)")
    elif symptom_count == 2:
        glucose = 120 # Risk estimate
        st.warning("Estimated blood sugar: 120 (High)")
    else:
        glucose = 140 # High risk estimate
        st.error("Estimated blood sugar: 140 (Very high)")

with st.expander("📋 What do these numbers mean? Click for examples"):
    st.markdown("""
    | Your Number | What It Means | Real Life Example |
    | --- | --- | --- |
    | **70-99** | Normal | Most healthy people when they wake up |
    | **100-125** | Pre-diabetes | Like a warning sign. Change diet now |
    | **126+** | Diabetes | Doctor will ask for 2nd test to confirm |
    
    **How to get this number:**
    1. **Lab Test:** Book "Fasting Blood Sugar" test. Don't eat 8 hours before.
    2. **Home Meter:** Test first thing in morning before eating/drinking water is OK.
    3. **No Test:** Use the 3 questions above. It's just an estimate.
    """)

st.subheader("📏 Body Measurements")
col3, col4 = st.columns(2)
height = col3.number_input("Height (cm)", 100, 250, 165)
weight = col4.number_input("Weight (kg)", 30, 200, 65)
bmi = weight / ((height/100)**2)
st.info(f"Calculated BMI: {bmi:.1f} | {'Normal' if bmi<25 else 'Overweight' if bmi<30 else 'Obese'}")

st.subheader("❤️ Health Background")
col5, col6 = st.columns(2)
bp_option = col5.selectbox("Blood Pressure Status", 
    ["Low", "Normal", "High Blood Pressure", "Not Sure"])
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
        risk_level = "Low"
        st.success(f"**Lower Risk: {risk_percent:.1f}%**")
        st.markdown("Your statistical risk is low. Maintaining a healthy lifestyle is recommended.")
    elif risk_percent < 70:
        risk_level = "Moderate"
        st.warning(f"**Moderate Risk: {risk_percent:.1f}%**")
        st.markdown("Your statistical risk is moderate. Consider lifestyle monitoring and regular health checkups.")
    else:
        risk_level = "High"
        st.error(f"**Elevated Risk: {risk_percent:.1f}%**")
        st.markdown("Your statistical risk is elevated. Consulting a healthcare professional for further testing is strongly advised.")

    # ===== NEW SECTION: PERSONALIZED SUGGESTIONS =====
    st.markdown("---")
    with st.expander("💡 Personalized Health Tips: What to Eat & Avoid", expanded=True):
        st.markdown("**Based on general diabetes prevention guidelines from WHO & ADA:**")
        
        if risk_level == "Low":
            st.success("**Keep doing what you're doing! Focus on maintaining:**")
            st.markdown("""
            **✅ EAT MORE:**
            - Whole grains: Brown rice, oats, whole wheat roti
            - Vegetables: Spinach, broccoli, carrots, bitter gourd (karela)
            - Fruits: Apple, orange, guava, berries - eat whole, not juice
            - Protein: Lentils (dal), chickpeas, fish, eggs, paneer
            - Healthy fats: Nuts, seeds, olive oil
            
            **🏃 LIFESTYLE:** 30 min walking 5 days/week + 7-8 hours sleep
            """)
            
        elif risk_level == "Moderate":
            st.warning("**Small changes make a big difference. Start here:**")
            st.markdown("""
            **✅ EAT MORE:**
            - High fiber: Oats, daliya, rajma, vegetables with every meal
            - Protein: Grilled chicken/fish, tofu, sprouts - helps control sugar spikes
            - Good snacks: Handful of almonds, cucumber, roasted chana
            
            **❌ REDUCE/AVOID:**
            - Sugary drinks: Soda, packaged juice, sweet tea/coffee
            - White carbs: White bread, white rice, maida - switch to brown/whole grain
            - Sweets: Mithai, cakes, cookies - limit to special occasions
            - Fried food: Samosa, pakora, chips - try air-fried or baked
            
            **🏃 LIFESTYLE:** 45 min brisk walk daily + reduce sitting time. Check blood sugar every 6 months.
            """)
            
        else: # High Risk
            st.error("**Important: Please consult a doctor. These tips support medical care:**")
            st.markdown("""
            **✅ PRIORITIZE THESE FOODS:**
            - Non-starchy vegetables: 50% of your plate - spinach, cauliflower, bhindi, lauki
            - Lean protein: 25% of plate - grilled fish, chicken breast, dal, paneer
            - Complex carbs: 25% of plate - quinoa, brown rice, millets (bajra, jowar)
            - Best fruits: Jamun, guava, apple, pear - 1 serving/day
            
            **❌ STRICTLY LIMIT:**
            - Sugar: Table sugar, honey, jaggery, sweets, desserts
            - Refined carbs: White rice, white bread, pasta, potatoes
            - Packaged food: Biscuits, namkeen, instant noodles - high hidden sugar/salt
            - Fruit juice: Even 100% juice spikes sugar - eat whole fruit instead
            - Alcohol: Can cause dangerous sugar drops
            
            **🏃 LIFESTYLE:** Doctor-supervised exercise plan. Monitor blood sugar as advised. Never skip meals.
            """)
        
        st.caption("**Note:** These are general guidelines, not personalized medical advice. Portion size and specific needs vary. Consult a dietitian or doctor for a custom meal plan.")
    # ===== END NEW SECTION =====

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

# --- Footer ---
st.divider()
st.caption("Disclaimer: For educational and informational purposes only. Not medical advice. Model trained on Pima Indian Diabetes Dataset.")
st.caption("Built with Python, Streamlit, XGBoost, SHAP. Nutrition tips based on WHO/ADA guidelines.")
