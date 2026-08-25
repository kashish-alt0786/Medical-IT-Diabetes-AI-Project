"""Reproducible training pipeline for Explainable AI Diabetes Risk Prediction.

The Pima dataset is fetched from a public raw source at training time. Missing
clinical measurements encoded as zero are imputed from the training fold only.
SMOTE is applied only to training data. Model selection prioritizes F1, Recall,
and ROC-AUC; accuracy is reported but is not the selection objective.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
TARGET = "Outcome"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data() -> pd.DataFrame:
    columns = FEATURES + [TARGET]
    df = pd.read_csv(DATA_URL, header=None, names=columns)
    if df.shape != (768, 9):
        raise ValueError(f"Unexpected dataset shape: {df.shape}; expected (768, 9)")
    for col in ZERO_AS_MISSING:
        df[col] = df[col].replace(0, np.nan)
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), FEATURES)
    ])


def make_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight=None, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=6, min_samples_leaf=3,
            random_state=42, n_jobs=-1, class_weight=None
        ),
        "XGBoost": XGBClassifier(
            n_estimators=350, max_depth=3, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85,
            reg_lambda=2.0, eval_metric="logloss", random_state=42,
            n_jobs=2, tree_method="hist"
        ),
    }


def evaluate(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    # Keep the standard 0.50 threshold for transparent, reproducible reporting.
    pred = (proba >= 0.50).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": float(accuracy_score(y_test, pred)),
        "Precision": float(precision_score(y_test, pred, zero_division=0)),
        "Recall": float(recall_score(y_test, pred, zero_division=0)),
        "F1": float(f1_score(y_test, pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, proba)),
        "False Negatives": int(fn),
        "False Positives": int(fp),
        "True Positives": int(tp),
        "True Negatives": int(tn),
    }


def main():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    results = {}
    fitted = {}
    for name, estimator in make_models().items():
        # Imputation is fitted only on X_train; SMOTE sees only transformed training data.
        pipe = Pipeline([
            ("preprocess", build_preprocessor()),
            ("smote", SMOTE(random_state=42, k_neighbors=5)),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        results[name] = evaluate(pipe, X_test, y_test)
        fitted[name] = pipe

    # Medical-screening-oriented hierarchy: F1 first, then Recall, then ROC-AUC.
    selected = max(results, key=lambda n: (
        results[n]["F1"], results[n]["Recall"], results[n]["ROC-AUC"]
    ))

    joblib.dump(fitted[selected], ROOT / "model.pkl")
    metrics_payload = {
        "dataset": "Pima Indians Diabetes Dataset",
        "dataset_source": DATA_URL,
        "samples": int(len(df)),
        "test_size": 0.20,
        "random_state": 42,
        "selection_priority": ["F1", "Recall", "ROC-AUC"],
        "selected_model": selected,
        "models": results,
    }
    (ROOT / "model_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    sha = os.getenv("GITHUB_SHA", "local")
    timestamp = datetime.now(timezone.utc).isoformat()
    meta = {
        "trained_at_utc": timestamp,
        "commit_sha": sha,
        "selected_model": selected,
        "accuracy": results[selected]["Accuracy"],
        "f1": results[selected]["F1"],
        "recall": results[selected]["Recall"],
        "roc_auc": results[selected]["ROC-AUC"],
        "dataset_samples": int(len(df)),
    }
    (ROOT / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    history_path = ROOT / "history.csv"
    row = pd.DataFrame([{**meta}])
    if history_path.exists():
        row.to_csv(history_path, mode="a", header=False, index=False)
    else:
        row.to_csv(history_path, index=False)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
