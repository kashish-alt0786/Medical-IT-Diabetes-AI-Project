import streamlit as st


def show_sidebar(t):
    with st.sidebar:
        st.markdown("# 🩺 Explainable AI Diabetes Risk Prediction")
        st.caption("Explainable Artificial Intelligence for Preventive Healthcare")
        st.success("Validation-first model workflow")
        st.divider()

        st.subheader("📊 Model Validation")
        st.info(
            "The upgraded workflow compares Logistic Regression, Random Forest and XGBoost. "
            "Models are selected using F1-score and Recall rather than changing accuracy cosmetically."
        )
        st.caption("Open **Model Metrics & Validation** from the Streamlit navigation to see the live held-out comparison.")
        st.divider()

        st.subheader("🧠 Machine Learning")
        st.markdown("""
**Model comparison**
- Logistic Regression
- Random Forest
- XGBoost

**Class balancing**
- SMOTE on training data only

**Explainability**
- SHAP

**Dataset**
- Pima Indians Diabetes Dataset

**Primary metrics**
- Recall
- F1-score
- ROC-AUC
""")
        st.divider()

        st.subheader("🥗 Manage Your Nutrition")
        st.info(
            "Want to take the next step after checking your diabetes risk? "
            "Use our connected nutrition app for educational meal and nutrition guidance."
        )
        st.link_button(
            "🥗 Manage Your Diet → NutriGuard-AI",
            "https://nutriguard-ai-rrzi6rnezvcba9dhtgzlrm.streamlit.app/",
            use_container_width=True,
        )
        st.caption(
            "General educational guidance only; it does not replace advice from a doctor or registered dietitian."
        )
        st.divider()

        st.subheader("🔗 Project")
        st.link_button(
            "💻 GitHub Repository",
            "https://github.com/kashish-alt0786/Medical-IT-Diabetes-AI-Project",
            use_container_width=True,
        )
        st.link_button(
            "🌐 Live Demo",
            "https://medical-it-diabetes-ai-project-jkv5wwfmjmjugk5frfffut.streamlit.app/",
            use_container_width=True,
        )
        st.divider()

        st.subheader("⚙ Technology Stack")
        st.markdown("""
- Python
- Streamlit
- XGBoost
- Random Forest
- Scikit-Learn
- imbalanced-learn / SMOTE
- SHAP
- Pandas / NumPy
""")
        st.divider()

        st.warning(
            "This tool is for educational and preventive screening purposes only. "
            "It does not diagnose diabetes and should never replace professional medical advice."
        )
