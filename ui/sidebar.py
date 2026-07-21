import streamlit as st


def show_sidebar(t):
    with st.sidebar:

        st.markdown("## 🩺 Diabetes Risk Predictor")

        st.caption("Explainable AI for Preventive Health Screening")

        st.divider()

        st.subheader("📊 AI Model")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Accuracy", "69.5%")

        with col2:
            st.metric("AUC", "0.76")

        st.metric("Recall", "67.3%")

        st.divider()

        st.subheader("🧠 Machine Learning")

        st.markdown("""
**Model**
- XGBoost Classifier

**Explainability**
- SHAP

**Dataset**
- Pima Indian Diabetes Dataset
""")

        st.divider()

        st.subheader("🔗 Project")

        st.markdown(
            "[💻 GitHub Repository](https://github.com/your-github-link)"
        )

        st.markdown(
            "[🌐 Live Demo](https://medical-it-diabetes-ai-project-jkv5wwfmjmjugk5frfffut.streamlit.app/)"
        )

        st.divider()

        st.info(
            "⚠️ This tool is intended for educational screening only and is not a medical diagnosis."
        )

        st.caption("Version 2.0")
