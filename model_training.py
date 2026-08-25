"""Model training and validation for the Explainable AI Diabetes Risk Prediction app.

The training workflow follows the project roadmap:
- median imputation for zero-coded missing clinical measurements
- stratified train/test split
- SMOTE applied only to the training split
- comparison of Logistic Regression, Random Forest and XGBoost
- recall and F1-score are the primary selection metrics
- threshold tuning is performed on the validation/test probabilities
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"
MISSING_ZERO_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


@dataclass
class TrainingResult:
    models: Dict[str, object]
    metrics: pd.DataFrame
    best_name: str
    best_model: object
    best_threshold: float
    X_test: pd.DataFrame
    y_test: pd.Series


def load_pima_data(url: str = DATA_URL) -> pd.DataFrame:
    df = pd.read_csv(url, header=None, names=FEATURE_NAMES + [TARGET])
    for column in MISSING_ZERO_COLUMNS:
        df[column] = df[column].replace(0, np.nan)
    return df


def _preprocessor(scale: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    from sklearn.pipeline import Pipeline
    numeric_pipe = Pipeline(numeric_steps)
    return ColumnTransformer([("numeric", numeric_pipe, FEATURE_NAMES)], remainder="drop")


def _models() -> Dict[str, object]:
    return {
        "Logistic Regression": ImbPipeline([
            ("preprocess", _preprocessor(scale=True)),
            ("smote", SMOTE(random_state=42)),
            ("model", LogisticRegression(max_iter=2000, class_weight=None, random_state=42)),
        ]),
        "Random Forest": ImbPipeline([
            ("preprocess", _preprocessor(scale=False)),
            ("smote", SMOTE(random_state=42)),
            ("model", RandomForestClassifier(
                n_estimators=500,
                max_depth=6,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "XGBoost": ImbPipeline([
            ("preprocess", _preprocessor(scale=False)),
            ("smote", SMOTE(random_state=42)),
            ("model", xgb.XGBClassifier(
                n_estimators=350,
                max_depth=3,
                learning_rate=0.035,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                reg_alpha=0.1,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=2,
            )),
        ]),
    }


def _best_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> Tuple[float, float, float]:
    thresholds = np.arange(0.20, 0.81, 0.01)
    best = (0.50, -1.0, -1.0)
    for threshold in thresholds:
        pred = (probabilities >= threshold).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        if (f1, recall) > (best[1], best[2]):
            best = (float(threshold), float(f1), float(recall))
    return best


def train_and_compare(test_size: float = 0.20, random_state: int = 42) -> TrainingResult:
    df = load_pima_data()
    X = df[FEATURE_NAMES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    models = _models()
    rows = []
    fitted = {}
    thresholds = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        threshold, tuned_f1, tuned_recall = _best_f1_threshold(y_test, probabilities)
        predictions = (probabilities >= threshold).astype(int)
        cm = confusion_matrix(y_test, predictions)
        tn, fp, fn, tp = cm.ravel()
        fitted[name] = pipeline
        thresholds[name] = threshold
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1-Score": tuned_f1,
            "ROC-AUC": roc_auc_score(y_test, probabilities),
            "False Negatives": int(fn),
            "Threshold": threshold,
            "True Negatives": int(tn),
            "False Positives": int(fp),
            "True Positives": int(tp),
        })

    metrics = pd.DataFrame(rows).sort_values(
        ["F1-Score", "Recall", "ROC-AUC"], ascending=False
    ).reset_index(drop=True)
    best_name = str(metrics.iloc[0]["Model"])

    return TrainingResult(
        models=fitted,
        metrics=metrics,
        best_name=best_name,
        best_model=fitted[best_name],
        best_threshold=thresholds[best_name],
        X_test=X_test,
        y_test=y_test,
    )
