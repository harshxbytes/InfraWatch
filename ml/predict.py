import joblib
import pandas as pd

from ml.feature_engineering import (
    create_features,
    FEATURE_COLUMNS
)


DELAY_MODEL_PATH = "models/delay_model.pkl"
COST_MODEL_PATH = "models/cost_model.pkl"
ANOMALY_MODEL_PATH = "models/anomaly_model.pkl"


delay_model = joblib.load(
    DELAY_MODEL_PATH
)

cost_model = joblib.load(
    COST_MODEL_PATH
)

anomaly_model = joblib.load(
    ANOMALY_MODEL_PATH
)


def risk_level(probability):

    if probability >= 0.70:
        return "High"

    if probability >= 0.40:
        return "Medium"

    return "Low"


def analyze_project(project):

    data = {
        "budget": project["budget"],
        "spent": project["spent"],
        "planned_progress": project["planned_progress"],
        "actual_progress": project["actual_progress"],
        "overdue_milestone_count": project[
            "overdue_milestones"
        ],
        "contractor_delay_history": project[
            "contractor_delay_history"
        ],

        "planned_duration_days": project.get(
            "planned_duration_days",
            1000
        ),

        "elapsed_days": project.get(
            "elapsed_days",
            600
        ),

        "change_orders": project.get(
            "change_orders",
            1
        )
    }

    df = pd.DataFrame(
        [data]
    )

    df = create_features(df)

    X = df[FEATURE_COLUMNS]

    delay_probability = float(
        delay_model.predict_proba(X)[0][1]
    )

    cost_probability = float(
        cost_model.predict_proba(X)[0][1]
    )

    anomaly_prediction = int(
        anomaly_model.predict(X)[0]
    )

    anomaly_score = float(
        -anomaly_model.score_samples(X)[0]
    )

    return {
        "project_id": project["project_id"],

        "delay_probability": round(
            delay_probability * 100,
            2
        ),

        "delay_risk": risk_level(
            delay_probability
        ),

        "cost_overrun_probability": round(
            cost_probability * 100,
            2
        ),

        "cost_overrun_risk": risk_level(
            cost_probability
        ),

        "anomaly_detected":
            anomaly_prediction == -1,

        "anomaly_score": round(
            anomaly_score,
            4
        )
    }