"""
train.py

Loads merged data from aqi_processed_full, trains 3 models
(Ridge, Random Forest, XGBoost) and logs each as a separate
MLflow run so you can compare them in the Dagshub UI.

What gets logged per run:
- Parameters  : model name, all hyperparams, features
- Metrics     : r2_fold per fold (step 1-5), r2_mean, r2_std, r2_min, r2_max
- Artifact    : trained pipeline (ready for serving)
- Tags        : dataset size, data source, city
"""

import os
import sys
# fix Windows terminal emoji crash from MLflow
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import psycopg2
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor

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

EXPERIMENT_NAME = "aqi-pm25-prediction"
mlflow.set_experiment(EXPERIMENT_NAME)

FEATURES = ["CO", "NH3", "NO2", "OZONE", "PM10", "SO2"]
TARGET   = "PM2.5"

# ── All 3 models with best params from grid search ────────────────────────────
MODELS = {
    "Ridge": {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler",  StandardScaler()),
            ("model",   Ridge(alpha=10.0)),
        ]),
        "params": {
            "model":            "Ridge",
            "alpha":            10.0,
            "imputer_strategy": "mean",
            "scaler":           "StandardScaler",
            "cv_folds":         5,
        },
        "register": False,   # not the best model
    },
    "RandomForest": {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler",  StandardScaler()),
            ("model",   RandomForestRegressor(
                n_estimators=500,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "params": {
            "model":              "RandomForestRegressor",
            "n_estimators":       500,
            "max_depth":          20,
            "min_samples_split":  5,
            "imputer_strategy":   "mean",
            "scaler":             "StandardScaler",
            "cv_folds":           5,
        },
        "register": True,    # best model → register in Model Registry
    },
    "XGBoost": {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler",  StandardScaler()),
            ("model",   XGBRegressor(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
                verbosity=0,
                n_jobs=-1,
            )),
        ]),
        "params": {
            "model":          "XGBRegressor",
            "n_estimators":   500,
            "max_depth":      7,
            "learning_rate":  0.05,
            "subsample":      0.8,
            "imputer_strategy": "mean",
            "scaler":         "StandardScaler",
            "cv_folds":       5,
        },
        "register": False,
    },
}


def load_data():
    with psycopg2.connect(DATABASE_URL) as conn:
        df = pd.read_sql(
            "SELECT * FROM aqi_processed_full ORDER BY last_update ASC",
            conn
        )
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"Loaded {len(df)} rows from aqi_processed_full")
    return df


def log_run(name, config, X, y, kf, dataset_rows):
    """Run CV, log everything to MLflow, optionally register model."""
    pipe   = config["pipeline"]
    params = config["params"]

    print(f"\n[{name}] Running 5-fold CV...")
    fold_scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")

    with mlflow.start_run(run_name=name):

        # params
        mlflow.log_param("features", ", ".join(FEATURES))
        mlflow.log_param("target",   TARGET)
        for k, v in params.items():
            mlflow.log_param(k, v)

        # fold scores — logged as steps so MLflow shows a graph
        for i, score in enumerate(fold_scores, 1):
            mlflow.log_metric("r2_fold", round(float(score), 4), step=i)

        # summary metrics
        mlflow.log_metric("r2_mean", round(float(fold_scores.mean()), 4))
        mlflow.log_metric("r2_std",  round(float(fold_scores.std()),  4))
        mlflow.log_metric("r2_min",  round(float(fold_scores.min()),  4))
        mlflow.log_metric("r2_max",  round(float(fold_scores.max()),  4))

        # tags
        mlflow.set_tag("dataset_rows", dataset_rows)
        mlflow.set_tag("data_source",  "aqi_processed_full")
        mlflow.set_tag("city",         "Delhi")

        # train on full data
        print(f"[{name}] Training on full dataset...")
        pipe.fit(X, y)

        # log model — register only the best one
        if config["register"]:
            mlflow.sklearn.log_model(
                pipe,
                name="model",
                registered_model_name="aqi-pm25-predictor"
            )
        else:
            mlflow.sklearn.log_model(pipe, name="model")

    print(f"[{name}] Mean R2={fold_scores.mean():.4f}  Std={fold_scores.std():.4f}  Folds={fold_scores.round(4)}")
    return fold_scores.mean()


def train():
    print("Loading data...")
    df = load_data()

    X = df[FEATURES]
    y = df[TARGET]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\nLogging all 3 models to MLflow...")
    print("="*55)

    summary = {}
    for name, config in MODELS.items():
        mean_r2 = log_run(name, config, X, y, kf, len(df))
        summary[name] = mean_r2

    print("\n" + "="*55)
    print("ALL RUNS COMPLETE")
    print("="*55)
    for name, r2 in summary.items():
        best = " <-- registered as aqi-pm25-predictor" if MODELS[name]["register"] else ""
        print(f"  {name:<20} Mean R2 = {r2:.4f}{best}")
    print("="*55)
    print(f"\nView all runs at: {DAGSHUB_TRACKING_URI}")


if __name__ == "__main__":
    train()
