import pandas as pd
import numpy as np


FEATURE_COLUMNS = [
    "budget",
    "spent",
    "planned_progress",
    "actual_progress",
    "progress_gap",
    "spend_ratio",
    "progress_efficiency",
    "spend_efficiency",
    "overdue_milestone_count",
    "contractor_delay_history",
    "planned_duration_days",
    "elapsed_days",
    "expenditure_velocity",
    "progress_velocity",
    "change_orders"
]


def create_features(df):

    df = df.copy()

    df["progress_gap"] = (
        df["planned_progress"]
        - df["actual_progress"]
    )

    df["spend_ratio"] = (
        df["spent"] /
        df["budget"].replace(0, np.nan)
    )

    df["progress_efficiency"] = (
        df["actual_progress"] /
        df["planned_progress"].replace(0, np.nan)
    )

    df["spend_efficiency"] = (
        df["spend_ratio"] /
        (df["actual_progress"] / 100).replace(0, np.nan)
    )

    df["expenditure_velocity"] = (
        df["spent"] /
        df["elapsed_days"].replace(0, np.nan)
    )

    df["progress_velocity"] = (
        df["actual_progress"] /
        df["elapsed_days"].replace(0, np.nan)
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.fillna(0)

    return df