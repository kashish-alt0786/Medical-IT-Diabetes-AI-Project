"""
config.py
---------------------------------------
Central configuration for the Diabetes Risk Predictor.
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

APP_NAME = "Diabetes Risk Predictor"
APP_VERSION = "1.0"
