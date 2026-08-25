import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]


def _load_json(filename):
    path = ROOT / filename
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def show_sidebar(t):
    with st.sidebar:
        st.markdown("# 🩺 Explainable AI Diabetes Risk Prediction")
        st.caption("Explainable Artificial Intelligence for Preventive Healthcare")
        st.success("Validation-first model workflow")
        st.divider()

        st.subheader("⚙ Automated MLOps")
        meta = _load_json("model_meta.json")
        if meta:
            trained = meta.get("trained_at_utc", "Unknown")
            commit = meta.get("commit_sha", "Unknown")
            selected = meta.get("selected_model", "Unknown")
            st.caption(f"Model automatically retrained on: **{trained}** via GitHub Actions.")
            st.caption(f"Selected model: **{selected}**")
            st.caption(f"Commit: `{commit[:7]}`")
        else:
            st.info("No successful automated training metadata is available yet. Run the GitHub Actions retraining workflow.")
        st.divider()

        st.subheader("📊 Model Validation")
        st.info(
            "The workflow compares Logistic Regression, Random Forest and XGBoost. "
            "Model selection prioritizes F1-score, Recall and ROC-AUC; accuracy remains a supporting metric."
        )
        st.caption("Open **Model Metrics & Validation** from the Streamlit navigation for the latest held-out comparison.")
        st.divider()

        st.subheader("🧠 Machine Learning")
        st.markdown("""
**Models**
- Logistic Regression
- Random Forest
- XGBoost

**Class balancing**
- SMOTE on training data only

**Explainability**
- SHAP

**Dataset**
- Pima Indians Diabetes Dataset (768 samples)

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
- GitHub Actions
""")
        st.divider()

        st.warning(
            "This tool is for educational and preventive screening purposes only. "
            "It does not diagnose diabetes and should never replace professional medical advice."
        )
