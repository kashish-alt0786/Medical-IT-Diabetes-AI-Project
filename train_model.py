import pandas as pd
import xgboost as xgb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv(...)
X = ...
y = ...
model = xgb.XGBClassifier(...)
model.fit(...)
joblib.dump(model, "model.pkl")
