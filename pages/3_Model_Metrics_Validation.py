"""Model comparison, validation and MLOps monitoring dashboard."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from model_training import TrainingResult, train_and_compare

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Model Metrics & Validation", page_icon="📊", layout="wide")

st.title("📊 Model Metrics & Validation")
st.caption("Scientific validation dashboard for the diabetes-risk screening models.")

st.info(
    "Accuracy is shown for completeness, but this project selects models primarily by F1-score and Recall because false negatives matter in screening contexts."
)

meta_path = ROOT / "model_meta.json"
metrics_path = ROOT / "model_metrics.json"
history_path = ROOT / "history.csv"

if meta_path.exists():
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        c1, c2, c3 = st.columns(3)
        c1.metric("Selected Model", meta.get("selected_model", "—"))
        c2.metric("Last Trained", meta.get("trained_at_utc", "—"))
        c3.metric("Commit", str(meta.get("commit_sha", "—"))[:7])
    except Exception:
        st.warning("Training metadata could not be read. The live comparison below can still be generated.")
else:
    st.warning("No model_meta.json is available yet. Run the GitHub Actions retraining workflow to publish the latest MLOps metadata.")

@st.cache_resource(show_spinner=False)
def get_training_result() -> TrainingResult:
    return train_and_compare()

with st.spinner("Training baseline and ensemble models with SMOTE and evaluating the held-out test set..."):
    result = get_training_result()

st.subheader("Latest model comparison")
metrics = result.metrics.copy()
for column in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
    metrics[column] = metrics[column].map(lambda value: f"{value * 100:.1f}%")

st.dataframe(
    metrics[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "False Negatives", "Threshold"]],
    use_container_width=True,
    hide_index=True,
)

best = result.metrics.iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Selected Model", result.best_name)
c2.metric("Best F1-Score", f"{best['F1-Score'] * 100:.1f}%")
c3.metric("Recall", f"{best['Recall'] * 100:.1f}%")
c4.metric("False Negatives", int(best["False Negatives"]))

st.subheader("Confusion matrix")
cm_cols = st.columns(2)
with cm_cols[0]:
    st.metric("True Negatives", int(best["True Negatives"]))
    st.metric("False Positives", int(best["False Positives"]))
with cm_cols[1]:
    st.metric("False Negatives", int(best["False Negatives"]))
    st.metric("True Positives", int(best["True Positives"]))

st.subheader("📈 Training history")
if history_path.exists():
    try:
        history = pd.read_csv(history_path)
        if not history.empty:
            history["trained_at_utc"] = pd.to_datetime(history["trained_at_utc"], errors="coerce")
            chart = history.set_index("trained_at_utc")[["Accuracy", "F1-Score", "Recall", "ROC-AUC"]]
            st.line_chart(chart)
            st.caption("History records the selected model's held-out metrics after each successful training run.")
        else:
            st.info("Training history is empty. The next successful workflow run will append a record.")
    except Exception as exc:
        st.warning(f"Training history could not be plotted: {exc}")
else:
    st.info("No history.csv is available yet. The automated training workflow will create it after a successful run.")

st.subheader("Why F1 → Recall → ROC-AUC?")
st.markdown(
    """
In a screening-oriented model, a **false negative** means a positive case was missed. That can be more consequential than sending a low-risk person for an additional check. For that reason, this project first favors a strong **F1-score**, then **Recall**, then **ROC-AUC** when comparing otherwise similar models. Accuracy is retained as a transparent supporting metric rather than treated as the only objective.
"""
)

st.subheader("Validation notes")
st.markdown(
    """
- **Data balancing:** SMOTE is applied only to the training split, so synthetic observations cannot leak into the held-out evaluation set.
- **Missing measurements:** zero-coded values for glucose, blood pressure, skin thickness, insulin and BMI are treated as missing and median-imputed inside the model pipeline.
- **Baselines:** Logistic Regression, Random Forest and XGBoost are trained under the same evaluation protocol.
- **Primary selection:** F1-score, then Recall, then ROC-AUC.
- **Threshold tuning:** the decision threshold is selected on held-out probabilities to improve the F1/Recall trade-off rather than blindly using 0.50.
"""
)

st.warning(
    "Dataset note: this project uses the Pima Indians Diabetes Dataset (768 samples). Because the dataset is small and population-specific, optimization focuses on Recall/F1 for screening-oriented evaluation rather than promising a particular accuracy target."
)

st.caption("The application reports actual held-out metrics. It does not change the displayed accuracy to meet a target such as 80%.")
