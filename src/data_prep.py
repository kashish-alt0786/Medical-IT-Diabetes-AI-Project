"""Reusable clinical data-preparation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def replace_clinical_zeros_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with zero-coded missing clinical measurements replaced by NaN."""
    cleaned = df.copy()
    for column in ZERO_AS_MISSING:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].replace(0, np.nan)
    return cleaned
