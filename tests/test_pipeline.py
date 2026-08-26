"""Automated CI quality checks for the diabetes-risk pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd

from model_training import MISSING_ZERO_COLUMNS, clean_missing_zeros
from src.inference_engine import load_model, predict_probability


def test_clinical_zero_values_become_missing():
    frame = pd.DataFrame({column: [0.0, 10.0] for column in MISSING_ZERO_COLUMNS})
    cleaned = clean_missing_zeros(frame)
    for column in MISSING_ZERO_COLUMNS:
        assert pd.isna(cleaned.loc[0, column])
        assert cleaned.loc[1, column] == 10.0


def test_model_probability_is_valid():
    model_path = Path("model.pkl")
    assert model_path.exists(), "model.pkl must be present before inference tests run"
    model = load_model()
    values = {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 79,
        "BMI": 28.0,
        "DiabetesPedigreeFunction": 0.35,
        "Age": 30,
    }
    probability = predict_probability(model, values)
    assert 0.0 <= probability <= 1.0
    assert np.isfinite(probability)
