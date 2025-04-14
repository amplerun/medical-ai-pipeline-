# FILE: modules/training/lab_xgboost.py

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def generate_fake_labs(n=1000):
    np.random.seed(42)
    X = np.random.rand(n, 10)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # simple binary logic
    return X, y

def train_lab_model():
    X, y = generate_fake_labs()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"✅ Lab XGBoost Accuracy: {acc:.4f}")

    joblib.dump(model, "models/lab_predictor.model")
    print("✅ Saved as 'lab_predictor.model'")

if __name__ == "__main__":
    train_lab_model()
