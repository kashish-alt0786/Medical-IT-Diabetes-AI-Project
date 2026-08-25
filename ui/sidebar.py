import streamlit as st


def show_sidebar(t):

    with st.sidebar:

        # ==========================================
        # BRANDING
        # ==========================================

        st.markdown("# 🩺 Explainable AI Diabetes Risk Prediction")

        st.caption(
            "Explainable Artificial Intelligence for Preventive Healthcare"
        )

        st.success("Version 4.0")

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
            "Current verified baseline. Model improvement is being evaluated using held-out validation rather than changing the reported score artificially."
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
        # CONNECTED NUTRITION APP
        # ==========================================

        st.subheader("🥗 Manage Your Nutrition")

        st.info(
            "Want to take the next step after checking your diabetes risk? "
            "Use our connected nutrition app for educational meal and nutrition guidance."
        )

        st.link_button(
            "🥗 Manage Your Diet → NutriGuard-AI",
            "https://nutriguard-ai-rrzi6rnezvcba9dhtgzlrm.streamlit.app/",
            use_container_width=True
        )

        st.caption(
            "The nutrition app provides general educational guidance and does not replace advice from a doctor or registered dietitian."
        )

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
