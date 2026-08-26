"""Cached model inference helpers."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model.pkl"
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


def load_model():
    """Load the serialized production classifier."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model.pkl is missing. Run the GitHub Actions MLOps workflow to generate it."
        )
    return joblib.load(MODEL_PATH)


def predict_probability(model, values: dict[str, float]) -> float:
    """Return a validated positive-class probability between 0 and 1."""
    frame = pd.DataFrame([values]).reindex(columns=FEATURE_NAMES)
    probability = float(model.predict_proba(frame)[0, 1])
    return max(0.0, min(1.0, probability))
