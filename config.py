"""
config.py
---------------------------------------
Central configuration for the Explainable AI Diabetes Risk Prediction System.
All project-wide constants are stored here.
"""

# ==========================================================
# Model Features (Must match training order)
# ==========================================================

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

# ==========================================================
# Default Values
# ==========================================================

DEFAULT_INSULIN = 80
DEFAULT_SKIN_THICKNESS = 20

# ==========================================================
# Risk Thresholds
# ==========================================================

LOW_RISK_THRESHOLD = 30
MODERATE_RISK_THRESHOLD = 65

# ==========================================================
# Input Validation
# ==========================================================

MIN_AGE = 1
MAX_AGE = 120

MIN_GLUCOSE = 40
MAX_GLUCOSE = 500

MIN_BMI = 10
MAX_BMI = 70

MIN_BP = 40
MAX_BP = 250

# ==========================================================
# BMI Categories
# ==========================================================

BMI_NORMAL = 25
BMI_OVERWEIGHT = 30

# ==========================================================
# Application Information
# ==========================================================

APP_NAME = "Explainable AI Diabetes Risk Prediction"
APP_VERSION = "4.0"
NUTRITION_APP_URL = "https://nutriguard-ai-rrzi6rnezvcba9dhtgzlrm.streamlit.app/"
