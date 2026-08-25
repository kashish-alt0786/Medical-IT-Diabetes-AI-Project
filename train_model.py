"""Train and compare the upgraded diabetes-risk models.

Run locally with:
    python train_model.py

The script prints a side-by-side validation table and saves the selected
model as model.pkl. The model is selected using F1-score, then Recall,
then ROC-AUC rather than optimizing accuracy alone.

It also writes model_meta.json and appends the latest validation metrics to
history.csv so the Streamlit app can expose the MLOps training state.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from model_training import train_and_compare


if __name__ == "__main__":
    result = train_and_compare()
    print("\nModel comparison on the held-out test set:\n")
    print(result.metrics.to_string(index=False))
    print(f"\nSelected model: {result.best_name}")
    print(f"Selected threshold: {result.best_threshold:.2f}")

    joblib.dump(result.best_model, "model.pkl")

    metrics = result.metrics.copy()
    metrics.to_json("model_metrics.json", orient="records", indent=2)
    Path("model_selection.json").write_text(
        json.dumps({
            "selected_model": result.best_name,
            "threshold": result.best_threshold,
            "selection_priority": ["F1-Score", "Recall", "ROC-AUC"],
        }, indent=2),
        encoding="utf-8",
    )

    # Persist reproducibility metadata for the Streamlit UI.
    trained_at = datetime.now(timezone.utc).isoformat()
    commit_sha = os.getenv("GITHUB_SHA", "local-run")
    best_row = result.metrics.iloc[0].to_dict()
    metadata = {
        "trained_at_utc": trained_at,
        "commit_sha": commit_sha,
        "selected_model": result.best_name,
        "threshold": float(result.best_threshold),
        "dataset": "Pima Indians Diabetes Dataset",
        "dataset_samples": 768,
        "selection_priority": ["F1-Score", "Recall", "ROC-AUC"],
        "metrics": [
            {
                key: (float(value) if hasattr(value, "item") else value)
                for key, value in row.items()
            }
            for row in result.metrics.to_dict(orient="records")
        ],
    }
    Path("model_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Append the selected model's held-out metrics for longitudinal monitoring.
    history_path = Path("history.csv")
    history_row = pd.DataFrame([{
        "trained_at_utc": trained_at,
        "commit_sha": commit_sha,
        "selected_model": result.best_name,
        "Accuracy": float(best_row["Accuracy"]),
        "Precision": float(best_row["Precision"]),
        "Recall": float(best_row["Recall"]),
        "F1-Score": float(best_row["F1-Score"]),
        "ROC-AUC": float(best_row["ROC-AUC"]),
        "False Negatives": int(best_row["False Negatives"]),
    }])
    if history_path.exists():
        history_row.to_csv(history_path, mode="a", header=False, index=False)
    else:
        history_row.to_csv(history_path, index=False)

    print("\nSaved model.pkl, model_metrics.json, model_selection.json, model_meta.json and history.csv")
