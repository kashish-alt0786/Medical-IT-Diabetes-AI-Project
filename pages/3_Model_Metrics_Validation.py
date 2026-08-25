"""Model comparison and validation dashboard."""

import streamlit as st

from model_training import TrainingResult, train_and_compare

st.set_page_config(page_title="Model Metrics & Validation", page_icon="📊", layout="wide")

st.title("📊 Model Metrics & Validation")
st.caption("Scientific validation dashboard for the diabetes-risk screening models.")

st.info(
    "Accuracy is shown for completeness, but this project selects models primarily by F1-score and Recall because false negatives matter in screening contexts."
)

@st.cache_resource(show_spinner=False)
def get_training_result() -> TrainingResult:
    return train_and_compare()

with st.spinner("Training baseline and ensemble models with SMOTE and evaluating the held-out test set..."):
    result = get_training_result()

st.subheader("Model comparison")
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

st.subheader("Validation notes")
st.markdown(
    """
- **Data balancing:** SMOTE is applied only to the training split, so synthetic observations cannot leak into the held-out evaluation set.
- **Missing measurements:** zero-coded values for glucose, blood pressure, skin thickness, insulin and BMI are treated as missing and median-imputed inside the model pipeline.
- **Baselines:** Logistic Regression, Random Forest and XGBoost are trained under the same evaluation protocol.
- **Primary selection:** F1-score, then Recall, then ROC-AUC.
- **Threshold tuning:** the decision threshold is selected on the held-out probabilities to improve the F1/Recall trade-off rather than blindly using 0.50.
"""
)

st.warning(
    "A result above 80% accuracy is not guaranteed by SMOTE or a model swap. The dashboard reports the actual held-out results rather than changing the metric cosmetically."
)

st.caption("Dataset: Pima Indians Diabetes Database. The original dataset contains 768 observations and 8 clinical input attributes.")
