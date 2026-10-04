import os
import sys

import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from ml.feature_engineering import (
    create_features,
    FEATURE_COLUMNS
)


DATA_PATH = "data/historical_projects.csv"
MODEL_DIR = "models"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


df = pd.read_csv(DATA_PATH)

df = create_features(df)

X = df[FEATURE_COLUMNS]


model = IsolationForest(
    n_estimators=300,
    contamination=0.08,
    random_state=42
)


model.fit(X)


model_path = os.path.join(
    MODEL_DIR,
    "anomaly_model.pkl"
)

joblib.dump(
    model,
    model_path
)


print("=" * 60)
print("ANOMALY DETECTION MODEL")
print("=" * 60)

print(
    f"Training samples: {len(X)}"
)

print(
    f"Model saved: {model_path}"
)