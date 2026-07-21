import streamlit as st


def show_sidebar(t):

    with st.sidebar:

        # ==========================================
        # LOGO
        # ==========================================

        st.markdown("# 🩺 Diabetes Risk Predictor")

        st.caption(
            "Explainable Artificial Intelligence for Preventive Healthcare"
        )

        st.success("Version 3.0")

        st.divider()

        # ==========================================
        # MODEL PERFORMANCE
        # ==========================================

        st.subheader("📊 AI Model Performance")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Accuracy",
                "69.5%"
            )

        with c2:
            st.metric(
                "AUC",
                "0.76"
            )

        st.metric(
            "Recall",
            "67.3%"
        )

        st.caption(
            "Optimized for screening (higher recall reduces missed high-risk patients)."
        )

        st.divider()

        # ==========================================
        # MODEL DETAILS
        # ==========================================

        st.subheader("🧠 Machine Learning")

        st.markdown("""
**Prediction Model**
- XGBoost Classifier

**Explainability**
- SHAP (Explainable AI)

**Training Dataset**
- Pima Indian Diabetes Dataset

**Hyperparameter Optimization**
- GridSearchCV

**Language Support**
- English
- हिन्दी
- 한국어
""")

        st.divider()

        # ==========================================
        # PROJECT LINKS
        # ==========================================

        st.subheader("🔗 Project")

        st.link_button(
            "💻 GitHub Repository",
            "https://github.com/kashish-alt0786/Medical-IT-Diabetes-AI-Project",
            use_container_width=True
        )

        st.link_button(
            "🌐 Live Demo",
            "https://medical-it-diabetes-ai-project-jkv5wwfmjmjugk5frfffut.streamlit.app/",
            use_container_width=True
        )

        st.divider()

        # ==========================================
        # TECHNOLOGY
        # ==========================================

        st.subheader("⚙ Technology Stack")

        st.markdown("""
- Python
- Streamlit
- XGBoost
- SHAP
- Scikit-Learn
- Pandas
- NumPy
""")

        st.divider()

        # ==========================================
        # DISCLAIMER
        # ==========================================

        st.warning(
            """
This tool is designed for **educational and preventive health screening only**.

It **does not diagnose diabetes** and should never replace professional medical advice.
"""
        )

        st.caption(
            "Developed as an Explainable AI Healthcare Project."
        )
