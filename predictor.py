import numpy as np
import pandas as pd


def _extract_estimator(model):
    """Return the final estimator from a normal or imbalanced-learn pipeline."""
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"]
    return model


def predict_risk(model, feature_names, pregnancies, glucose, bp, skin, insulin, bmi, dpf, age):
    input_dict = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    input_df = pd.DataFrame([input_dict]).reindex(columns=feature_names)
    proba = float(model.predict_proba(input_df)[0][1])
    risk_percent = round(proba * 100, 1)

    if risk_percent < 30:
        risk_level, color = "Low", "green"
    elif risk_percent < 65:
        risk_level, color = "Moderate", "orange"
    else:
        risk_level, color = "High", "red"

    # Global model importance for a transparent fallback explanation.
    # Patient-specific explanations are provided by the SHAP page.
    estimator = _extract_estimator(model)
    reasons = []
    try:
        if hasattr(estimator, "feature_importances_"):
            scores = np.asarray(estimator.feature_importances_, dtype=float)
        elif hasattr(estimator, "coef_"):
            scores = np.abs(np.asarray(estimator.coef_[0], dtype=float))
        else:
            scores = np.zeros(len(feature_names))

        total = scores.sum()
        if total > 0:
            scores = scores / total
        ranked = sorted(zip(feature_names, scores), key=lambda item: item[1], reverse=True)
        reasons = [(name, float(score)) for name, score in ranked[:3]]
    except Exception:
        reasons = []

    return risk_percent, risk_level, color, input_df, reasons
