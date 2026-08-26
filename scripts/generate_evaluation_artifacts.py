"""Generate reproducible evaluation images and a local copy of the Pima CSV."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, auc

from model_training import load_pima_data, train_and_compare

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DATA = ROOT / "data"
ASSETS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)


def main() -> None:
    raw = pd.read_csv(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        header=None,
        names=[
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
        ],
    )
    raw.to_csv(DATA / "pima_indians_diabetes.csv", index=False)

    result = train_and_compare()
    best = result.best_model
    estimator = best.named_steps["model"]
    preprocessor = best.named_steps["preprocess"]
    transformed = preprocessor.transform(result.X_test)
    probabilities = best.predict_proba(result.X_test)[:, 1]
    predictions = (probabilities >= result.best_threshold).astype(int)

    ConfusionMatrixDisplay.from_predictions(
        result.y_test,
        predictions,
        display_labels=["Non-diabetic", "Diabetic"],
        cmap="Blues",
        colorbar=False,
    )
    plt.title(f"Confusion Matrix — {result.best_name}")
    plt.tight_layout()
    plt.savefig(ASSETS / "confusion_matrix.png", dpi=220, bbox_inches="tight")
    plt.close()

    fpr, tpr, _ = roc_curve(result.y_test, probabilities)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"{result.best_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ASSETS / "roc_auc_curve.png", dpi=220, bbox_inches="tight")
    plt.close()

    feature_names = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
    ]
    try:
        explainer = shap.Explainer(estimator, transformed)
        explanation = explainer(transformed)
        values = np.abs(explanation.values).mean(axis=0)
    except Exception:
        values = np.abs(getattr(estimator, "feature_importances_", np.ones(len(feature_names))))

    order = np.argsort(values)
    plt.figure(figsize=(8, 5))
    plt.barh(np.array(feature_names)[order], np.array(values)[order])
    plt.xlabel("Mean absolute contribution")
    plt.title("Global Feature Importance via Explainable AI")
    plt.tight_layout()
    plt.savefig(ASSETS / "shap_importance.png", dpi=220, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
