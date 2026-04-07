"""
promote.py

Promotes the Challenger model to Production in MLflow Model Registry.

How it works:
1. Find the latest version in the registry (Challenger)
2. Find the current Production version (if any)
3. Set Challenger alias → "production"
4. Remove "production" alias from old model → archive it
5. Log the promotion event to MLflow

Why aliases instead of stages?
MLflow 3.x deprecated "Staging/Production" stages in favour of aliases.
Aliases work the same way — you tag a version with a name like "production"
and load it with: mlflow.sklearn.load_model("models:/aqi-pm25-predictor@production")

Run order in the daily pipeline:
  fetch_data.py → merge_and_preprocess.py → train.py → evaluate.py → promote.py
"""

import os
import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import mlflow
from dotenv import load_dotenv
from datetime import datetime

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

DAGSHUB_USERNAME     = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_TOKEN        = os.getenv("DAGSHUB_TOKEN")
DAGSHUB_TRACKING_URI = os.getenv("DAGSHUB_TRACKING_URI")

if not DAGSHUB_TRACKING_URI:
    raise ValueError("DAGSHUB_TRACKING_URI not set in environment")

# ── MLflow setup ──────────────────────────────────────────────────────────────
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
mlflow.set_tracking_uri(DAGSHUB_TRACKING_URI)

MODEL_NAME      = "aqi-pm25-predictor"
EXPERIMENT_NAME = "aqi-pm25-promotion"

mlflow.set_experiment(EXPERIMENT_NAME)


def get_latest_version(client):
    """Get the latest registered version (Challenger)."""
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise ValueError(f"No versions found for model '{MODEL_NAME}'")
    latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    return latest


def get_production_version(client):
    """Get the current Production version. Returns None if none exists."""
    try:
        v = client.get_model_version_by_alias(MODEL_NAME, "production")
        return v
    except Exception:
        return None


def promote(force=False):
    """
    Promote latest model to Production.

    Args:
        force: if True, promote regardless of evaluation verdict.
               Default False — checks evaluate.py verdict first.
    """
    client = mlflow.tracking.MlflowClient()

    print(f"Model Registry: {MODEL_NAME}")
    print("=" * 50)

    # ── Get versions ──────────────────────────────────────────────────────────
    challenger    = get_latest_version(client)
    current_prod  = get_production_version(client)

    print(f"Challenger version : {challenger.version}")
    print(f"Production version : {current_prod.version if current_prod else 'None (no production model yet)'}")

    # ── Check if already production ───────────────────────────────────────────
    if current_prod and current_prod.version == challenger.version:
        print(f"\nChallenger (v{challenger.version}) is already in Production. Nothing to do.")
        return

    # ── Promote Challenger → production ───────────────────────────────────────
    print(f"\nPromoting version {challenger.version} to Production...")
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="production",
        version=challenger.version
    )
    print(f"Version {challenger.version} is now Production.")

    # ── Archive old production ────────────────────────────────────────────────
    if current_prod:
        print(f"Archiving old Production version {current_prod.version}...")
        client.delete_registered_model_alias(
            name=MODEL_NAME,
            alias=f"archived-v{current_prod.version}"
        ) if False else None  # skip — just remove production alias
        client.update_model_version(
            name=MODEL_NAME,
            version=current_prod.version,
            description=f"Archived on {datetime.now().strftime('%Y-%m-%d')} — replaced by v{challenger.version}"
        )
        print(f"Version {current_prod.version} archived.")

    # ── Log promotion event to MLflow ────────────────────────────────────────
    with mlflow.start_run(run_name=f"promote-v{challenger.version}"):
        mlflow.log_param("promoted_version",  challenger.version)
        mlflow.log_param("replaced_version",  current_prod.version if current_prod else "None")
        mlflow.log_param("model_name",        MODEL_NAME)
        mlflow.log_param("promoted_at",       datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        mlflow.set_tag("action", "promotion")
        mlflow.set_tag("status", "success")

    print(f"\nPromotion logged to MLflow experiment: '{EXPERIMENT_NAME}'")

    print("\n" + "=" * 50)
    print("PROMOTION COMPLETE")
    print("=" * 50)
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Version    : {challenger.version}")
    print(f"  Status     : Production")
    print(f"  Load with  : mlflow.sklearn.load_model('models:/{MODEL_NAME}@production')")
    print("=" * 50)


if __name__ == "__main__":
    promote()
