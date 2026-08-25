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


def _normalise_metrics(raw_metrics, meta):
    """Support list-format and dictionary-format validation files."""
    if isinstance(raw_metrics, list):
        models = {}
        for i, row in enumerate(raw_metrics):
            if isinstance(row, dict):
                name = row.get("Model") or f"Model {i + 1}"
                models[str(name)] = row
        return (meta or {}).get("selected_model"), models

    if isinstance(raw_metrics, dict):
        raw_models = raw_metrics.get("models")
        if isinstance(raw_models, dict):
            models = {
                str(name): row
                for name, row in raw_models.items()
                if isinstance(row, dict)
            }
            return raw_metrics.get("selected_model") or (meta or {}).get("selected_model"), models
        if isinstance(raw_models, list):
            models = {}
            for i, row in enumerate(raw_models):
                if isinstance(row, dict):
                    name = row.get("Model") or f"Model {i + 1}"
                    models[str(name)] = row
            return raw_metrics.get("selected_model") or (meta or {}).get("selected_model"), models

    if isinstance(meta, dict) and isinstance(meta.get("metrics"), list):
        models = {}
        for i, row in enumerate(meta["metrics"]):
            if isinstance(row, dict):
                name = row.get("Model") or f"Model {i + 1}"
                models[str(name)] = row
        return meta.get("selected_model"), models

    return None, {}


def _metric_value(row, *keys):
    for key in keys:
        value = row.get(key) if isinstance(row, dict) else None
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def show_sidebar(t):
    with st.sidebar:
        st.markdown("# 🩺 Explainable AI Diabetes Risk Prediction")
        st.caption("Explainable Artificial Intelligence for Preventive Healthcare")
        st.success("Validation-first model workflow")
        st.divider()

        st.subheader("⚙ Automated MLOps")
        meta = _load_json("model_meta.json")
        if isinstance(meta, dict):
            trained = meta.get("trained_at_utc", "Unknown")
            commit = meta.get("commit_sha", "Unknown")
            selected_meta = meta.get("selected_model", "Unknown")
            st.caption(f"Model automatically retrained on: **{trained}** via GitHub Actions.")
            st.caption(f"Selected model: **{selected_meta}**")
            st.caption(f"Commit: `{str(commit)[:7]}`")
        else:
            st.info("No successful automated training metadata is available yet. Run the GitHub Actions retraining workflow.")
        st.divider()

        st.subheader("📊 Model Validation")
        raw_metrics = _load_json("model_metrics.json")
        selected, models = _normalise_metrics(raw_metrics, meta)

        if models:
            if not selected or selected not in models:
                selected = next(iter(models))

            selected_metrics = models.get(selected, {})
            st.success(f"Selected model: **{selected}**")

            cols = st.columns(2)
            cols[0].metric("Accuracy", f"{_metric_value(selected_metrics, 'Accuracy') * 100:.1f}%")
            cols[1].metric("F1-Score", f"{_metric_value(selected_metrics, 'F1-Score', 'F1') * 100:.1f}%")
            cols = st.columns(2)
            cols[0].metric("Recall", f"{_metric_value(selected_metrics, 'Recall') * 100:.1f}%")
            cols[1].metric("ROC-AUC", f"{_metric_value(selected_metrics, 'ROC-AUC'):.2f}")

            # Build the comparison table without relying on a DataFrame index.
            # This avoids a pandas KeyError if a future metrics artifact changes its columns.
            comparison_rows = []
            for model_name, row in models.items():
                if not isinstance(row, dict):
                    continue
                comparison_rows.append({
                    "Model": str(model_name),
                    "Accuracy": _metric_value(row, "Accuracy"),
                    "F1-Score": _metric_value(row, "F1-Score", "F1"),
                    "Recall": _metric_value(row, "Recall"),
                    "ROC-AUC": _metric_value(row, "ROC-AUC"),
                })

            if comparison_rows:
                comparison = pd.DataFrame(comparison_rows, columns=["Model", "Accuracy", "F1-Score", "Recall", "ROC-AUC"])
                st.dataframe(comparison.round(3), use_container_width=True, hide_index=True)

            st.caption(
                "Evaluation uses an untouched 20% held-out test set. "
                "Model selection prioritizes F1-score → Recall → ROC-AUC; accuracy is reported as a supporting metric."
            )
            st.warning(
                "Dataset note: the Pima Indians Diabetes Dataset contains 768 samples. "
                "Because of its limited size, this project emphasizes Recall and F1-score for screening rather than optimizing accuracy alone."
            )
        else:
            st.info("Validation metrics will appear here after a successful GitHub Actions training run.")
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
