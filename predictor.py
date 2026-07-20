import pandas as pd
import numpy as np
import xgboost as xgb

def predict_risk(model, feature_names, pregnancies, glucose, bp, skin, insulin, bmi, dpf, age):
    # Build dataframe in correct order
    input_dict = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skin,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    # Ensure order matches training
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_names)

    # SAFE prediction - works with both xgboost 1.7.6 and 2.x
    try:
        proba = model.predict_proba(input_df)[0][1]
    except Exception:
        # Fallback for version mismatch
        dmatrix = xgb.DMatrix(input_df)
        booster = model.get_booster() if hasattr(model, 'get_booster') else model
        proba = booster.predict(dmatrix)[0]

    risk_percent = round(float(proba * 100), 1)

    if risk_percent < 30:
        risk_level = "Low"
        color = "green"
    elif risk_percent < 65:
        risk_level = "Moderate"
        color = "orange"
    else:
        risk_level = "High"
        color = "red"

    # Top reasons for Phase 2 (simple, fast, no SHAP heavy calc)
    # Use feature values as reasons
    reasons = []
    if glucose >= 126:
        reasons.append(("Glucose", 0.45))
    if bmi >= 30:
        reasons.append(("BMI", 0.35))
    if age >= 45:
        reasons.append(("Age", 0.20))
    if dpf > 0.5:
        reasons.append(("Family History", 0.25))
    if bp >= 90:
        reasons.append(("BloodPressure", 0.15))

    if not reasons:
        reasons = [("Glucose", 0.1), ("BMI", 0.05), ("Age", 0.02)]

    return risk_percent, risk_level, color, input_df, reasons
