"""
evaluate.py

Compares the latest trained model (Challenger) against the current
Production model from MLflow registry.

How it works:
1. Load Production model from MLflow registry (by stage="Production")
2. Load Challenger model (latest version in registry)
3. Split data into train/test using TIME — last 20% of dates = test set
   (we use time-based split, not random, because AQI is time-series data)
4. Score both models on the same test set
5. Log comparison to MLflow
6. Print verdict — should we promote or not?

Why time-based split?
- Random split would let the model "see" future data during training
- In production, the model always predicts future readings it has never seen
- So we test it the same way: train on older data, test on newer data
"""

import os
import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import psycopg2
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

DATABASE_URL         = os.getenv("DATABASE_URL")
DAGSHUB_USERNAME     = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_TOKEN        = os.getenv("DAGSHUB_TOKEN")
DAGSHUB_TRACKING_URI = os.getenv("DAGSHUB_TRACKING_URI")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set in environment")
if not DAGSHUB_TRACKING_URI:
    raise ValueError("DAGSHUB_TRACKING_URI not set in environment")

# ── MLflow setup ──────────────────────────────────────────────────────────────
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
mlflow.set_tracking_uri(DAGSHUB_TRACKING_URI)

EXPERIMENT_NAME  = "aqi-pm25-evaluation"
MODEL_NAME       = "aqi-pm25-predictor"
FEATURES         = ["CO", "NH3", "NO2", "OZONE", "PM10", "SO2"]
TARGET           = "PM2.5"
PROMOTE_THRESHOLD = 0.01   # new model must beat production by at least 0.01 R²

mlflow.set_experiment(EXPERIMENT_NAME)


def load_data():
    """Load full processed dataset ordered by date."""
    with psycopg2.connect(DATABASE_URL) as conn:
        df = pd.read_sql(
            "SELECT * FROM aqi_processed_full ORDER BY last_update ASC",
            conn
        )
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"Loaded {len(df)} rows")
    return df


def time_based_split(df, test_ratio=0.2):
    """
    Split by time — oldest rows = train, newest rows = test.
    This simulates real production: model trained on past, tested on future.
    """
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]
    print(f"Train: {len(train)} rows ({train['last_update'].min()} to {train['last_update'].max()})")
    print(f"Test : {len(test)} rows  ({test['last_update'].min()} to {test['last_update'].max()})")
    return train, test


def load_production_model(client):
    """Load current Production model. Returns None if no Production model exists."""
    try:
        versions = client.get_model_version_by_alias(MODEL_NAME, "production")
        model_uri = f"models:/{MODEL_NAME}@production"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Production model loaded: {MODEL_NAME} version {versions.version}")
        return model, versions.version
    except Exception:
        print("No Production model found in registry yet.")
        return None, None


def load_challenger_model(client):
    """Load the latest registered version as the Challenger."""
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise ValueError(f"No versions found for model '{MODEL_NAME}'")
    # get latest version by version number
    latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    model_uri = f"models:/{MODEL_NAME}/{latest.version}"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Challenger model loaded: {MODEL_NAME} version {latest.version}")
    return model, latest.version


def score_model(model, X_test, y_test, label):
    """Score a model and return metrics dict."""
    y_pred = model.predict(X_test)
    metrics = {
        "r2":   round(float(r2_score(y_test, y_pred)), 4),
        "mae":  round(float(mean_absolute_error(y_test, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
    }
    print(f"  {label:<20} R2={metrics['r2']}  MAE={metrics['mae']}  RMSE={metrics['rmse']}")
    return metrics, y_pred


def evaluate():
    print("Loading data...")
    df = load_data()

    print("\nTime-based train/test split (80/20):")
    train, test = time_based_split(df)

    X_test = test[FEATURES]
    y_test = test[TARGET]

    client = mlflow.tracking.MlflowClient()

    print("\nLoading models from MLflow registry...")
    prod_model, prod_version     = load_production_model(client)
    challenger, challenger_version = load_challenger_model(client)

    print("\nScoring on test set:")
    print("-" * 55)
    challenger_metrics, _ = score_model(challenger, X_test, y_test, f"Challenger (v{challenger_version})")

    if prod_model is not None:
        prod_metrics, _ = score_model(prod_model, X_test, y_test, f"Production (v{prod_version})")
    else:
        prod_metrics = {"r2": None, "mae": None, "rmse": None}
        print(f"  Production            No model yet — Challenger wins by default")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("-" * 55)
    if prod_metrics["r2"] is None:
        verdict   = "PROMOTE"
        reason    = "No production model exists yet"
    elif challenger_metrics["r2"] >= prod_metrics["r2"] + PROMOTE_THRESHOLD:
        verdict   = "PROMOTE"
        reason    = f"Challenger R2 ({challenger_metrics['r2']}) beats Production R2 ({prod_metrics['r2']}) by {challenger_metrics['r2'] - prod_metrics['r2']:.4f}"
    else:
        verdict   = "KEEP CURRENT"
        reason    = f"Challenger R2 ({challenger_metrics['r2']}) does not beat Production R2 ({prod_metrics['r2']}) by threshold ({PROMOTE_THRESHOLD})"

    print(f"\nVerdict : {verdict}")
    print(f"Reason  : {reason}")

    # ── Log evaluation run to MLflow ──────────────────────────────────────────
    with mlflow.start_run(run_name=f"evaluation-v{challenger_version}"):
        mlflow.log_param("challenger_version", challenger_version)
        mlflow.log_param("production_version", prod_version)
        mlflow.log_param("test_rows",          len(test))
        mlflow.log_param("promote_threshold",  PROMOTE_THRESHOLD)

        mlflow.log_metric("challenger_r2",   challenger_metrics["r2"])
        mlflow.log_metric("challenger_mae",  challenger_metrics["mae"])
        mlflow.log_metric("challenger_rmse", challenger_metrics["rmse"])

        if prod_metrics["r2"] is not None:
            mlflow.log_metric("production_r2",   prod_metrics["r2"])
            mlflow.log_metric("production_mae",  prod_metrics["mae"])
            mlflow.log_metric("production_rmse", prod_metrics["rmse"])
            mlflow.log_metric("r2_improvement",  round(challenger_metrics["r2"] - prod_metrics["r2"], 4))

        mlflow.set_tag("verdict", verdict)
        mlflow.set_tag("reason",  reason)

    print(f"\nEvaluation logged to MLflow experiment: '{EXPERIMENT_NAME}'")
    return verdict, challenger_metrics, prod_metrics


if __name__ == "__main__":
    verdict, challenger_metrics, prod_metrics = evaluate()

    print("\n" + "=" * 55)
    print("EVALUATION SUMMARY")
    print("=" * 55)
    print(f"  Challenger R2  : {challenger_metrics['r2']}")
    print(f"  Production R2  : {prod_metrics['r2']}")
    print(f"  Verdict        : {verdict}")
    print("=" * 55)

    if verdict == "PROMOTE":
        print("\nNext step: run promote.py to move Challenger to Production")
    else:
        print("\nNext step: collect more data or tune model further")
