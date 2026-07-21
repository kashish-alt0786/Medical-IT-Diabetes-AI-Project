import streamlit as st


def show_sidebar(t):
    with st.sidebar:

        st.markdown(f"## {t['title']}")
        st.caption(t["subtitle"])

        st.divider()

        st.subheader(t["model_perf"])

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                t["accuracy"],
                "69.5%",
                help=t["accuracy_desc"]
            )

        with col2:
            st.metric(
                t["auc"],
                "0.76",
                help=t["auc_desc"]
            )

        st.metric(
            t["recall"],
            "67.3%",
            help=t["recall_desc"]
        )

        st.divider()

        st.subheader("🧠 AI Model")

        st.markdown("""
**Algorithm**
- XGBoost Classifier

**Explainability**
- SHAP

**Dataset**
- Pima Indian Diabetes Dataset

**Deployment**
- Streamlit Cloud
""")

        st.divider()

        st.subheader(t["project_links"])

        st.markdown(
            """
💻 **GitHub**

https://github.com/kashish-alt0786/Medical-IT-Diabetes-AI-Project
"""
        )

        st.markdown(
            """
🌐 **Live Demo**

https://medical-it-diabetes-ai-project-jkv5wwfmjmjugk5frfffut.streamlit.app/
"""
        )

        st.divider()

        st.subheader(t["tech_stack"])

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

        st.warning(t["disclaimer"])

        st.caption("Version 3.0")
