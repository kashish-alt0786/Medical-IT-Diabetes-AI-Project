"""Train and compare the upgraded diabetes-risk models.

Run locally with:
    python train_model.py

The script prints a side-by-side validation table and saves the selected
model as model.pkl. The model is selected using F1-score, then Recall,
then ROC-AUC rather than optimizing accuracy alone.
"""

import json
from pathlib import Path

import joblib

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

    print("\nSaved model.pkl, model_metrics.json and model_selection.json")
