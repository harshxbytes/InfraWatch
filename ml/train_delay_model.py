import os
import sys

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier

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

y = df["delayed"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)


model.fit(
    X_train,
    y_train
)


predictions = model.predict(X_test)

probabilities = model.predict_proba(
    X_test
)[:, 1]


accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


print("\n" + "=" * 60)
print("DELAY PREDICTION MODEL")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    predictions
))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)


model_path = os.path.join(
    MODEL_DIR,
    "delay_model.pkl"
)

joblib.dump(
    model,
    model_path
)


feature_path = os.path.join(
    MODEL_DIR,
    "feature_columns.pkl"
)

joblib.dump(
    FEATURE_COLUMNS,
    feature_path
)


print("\nModel saved:")
print(model_path)