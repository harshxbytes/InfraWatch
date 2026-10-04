import os
import numpy as np
import pandas as pd

np.random.seed(42)

N = 6000

os.makedirs("data", exist_ok=True)

project_types = [
    "Road",
    "Bridge",
    "Hospital",
    "Water",
    "Railway",
    "Energy",
    "Urban Development"
]

departments = [
    "Public Works",
    "Health Infrastructure",
    "Urban Development",
    "Transport",
    "Energy",
    "Water Resources"
]

states = [
    "Madhya Pradesh",
    "Maharashtra",
    "Tamil Nadu",
    "Rajasthan",
    "Gujarat",
    "Karnataka",
    "Uttar Pradesh",
    "Telangana",
    "Odisha",
    "West Bengal"
]

df = pd.DataFrame()

df["project_id"] = [
    f"HP{100000 + i}"
    for i in range(N)
]

df["project_type"] = np.random.choice(
    project_types,
    N
)

df["department"] = np.random.choice(
    departments,
    N
)

df["state"] = np.random.choice(
    states,
    N
)

df["budget"] = np.random.uniform(
    20_000_000,
    500_000_000,
    N
)

df["planned_duration_days"] = np.random.randint(
    180,
    1800,
    N
)

df["elapsed_days"] = (
    df["planned_duration_days"] *
    np.random.uniform(0.2, 1.05, N)
).astype(int)

df["planned_progress"] = (
    df["elapsed_days"] /
    df["planned_duration_days"] *
    100
)

df["planned_progress"] = df[
    "planned_progress"
].clip(5, 95)

delay_factor = np.random.normal(
    0,
    8,
    N
)

df["actual_progress"] = (
    df["planned_progress"]
    - delay_factor
)

df["actual_progress"] = df[
    "actual_progress"
].clip(0, 100)

df["progress_gap"] = (
    df["planned_progress"]
    - df["actual_progress"]
)

df["overdue_milestone_count"] = np.random.poisson(
    1.5,
    N
)

df["overdue_milestone_count"] = df[
    "overdue_milestone_count"
].clip(0, 8)

df["contractor_delay_history"] = np.random.poisson(
    1,
    N
)

df["contractor_delay_history"] = df[
    "contractor_delay_history"
].clip(0, 5)

progress_ratio = (
    df["actual_progress"] / 100
)

base_spending = (
    df["budget"] *
    progress_ratio
)

spending_noise = np.random.normal(
    0,
    0.08,
    N
)

df["spent"] = (
    base_spending *
    (1 + spending_noise)
)

df["spent"] = df["spent"].clip(
    1_000_000,
    None
)

df["spend_ratio"] = (
    df["spent"] /
    df["budget"]
)

df["progress_efficiency"] = (
    df["actual_progress"] /
    df["planned_progress"].replace(0, 1)
)

df["spend_efficiency"] = (
    df["spend_ratio"] /
    (df["actual_progress"] / 100).replace(0, 0.01)
)

df["expenditure_velocity"] = (
    df["spent"] /
    df["elapsed_days"].replace(0, 1)
)

df["progress_velocity"] = (
    df["actual_progress"] /
    df["elapsed_days"].replace(0, 1)
)

df["change_orders"] = np.random.poisson(
    1,
    N
)

df["change_orders"] = df[
    "change_orders"
].clip(0, 6)

# --------------------------------------------------
# Future delay outcome
# --------------------------------------------------

risk_signal = (
    0.08 * df["progress_gap"]
    + 0.45 * df["overdue_milestone_count"]
    + 0.55 * df["contractor_delay_history"]
    + 4.0 * np.maximum(
        df["spend_ratio"] -
        df["actual_progress"] / 100,
        0
    )
    + 0.35 * df["change_orders"]
    + np.random.normal(0, 2, N)
)

delay_probability = (
    1 /
    (
        1 +
        np.exp(
            -(risk_signal - 3.5) / 3
        )
    )
)

df["delayed"] = (
    np.random.random(N)
    < delay_probability
).astype(int)

# --------------------------------------------------
# Future cost overrun
# --------------------------------------------------

cost_signal = (
    1.2 * np.maximum(
        df["spend_ratio"] -
        df["actual_progress"] / 100,
        0
    )
    + 0.08 * df["change_orders"]
    + 0.05 * df["contractor_delay_history"]
    + np.random.normal(0, 0.15, N)
)

cost_probability = (
    1 /
    (
        1 +
        np.exp(
            -(cost_signal - 0.15) * 8
        )
    )
)

df["cost_overrun"] = (
    np.random.random(N)
    < cost_probability
).astype(int)

# --------------------------------------------------
# Save
# --------------------------------------------------

output = "data/historical_projects.csv"

df.to_csv(
    output,
    index=False
)

print("=" * 60)
print("DATASET CREATED")
print("=" * 60)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nDelay distribution:")
print(df["delayed"].value_counts())

print("\nCost overrun distribution:")
print(df["cost_overrun"].value_counts())

print(f"\nSaved to: {output}")