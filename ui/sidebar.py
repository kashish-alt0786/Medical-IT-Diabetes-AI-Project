import streamlit as st


def show_sidebar(t):
    with st.sidebar:
        st.header(t["model_perf"])

        st.metric(t["recall"], "67.3%", t["recall_desc"])
        st.metric(t["auc"], "0.76", t["auc_desc"])
        st.metric(t["accuracy"], "69.5%", t["accuracy_desc"])

        st.markdown("---")

        st.markdown(f"**{t['project_links']}**")
        st.markdown(
            "[💻 GitHub Code](https://github.com/kashish-alt0786/Medical-IT-Diabetes-AI-Project)"
        )
        st.markdown(
            "[Kaggle Notebook](https://www.kaggle.com/code/kashish0000000/explainable-ai-diabetes-risk-prediction)"
        )

        st.markdown("---")

        st.markdown(
            f"**{t['tech_stack']}** `Python` `XGBoost` `SHAP` `Streamlit`"
        )
