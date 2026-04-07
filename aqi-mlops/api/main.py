"""
main.py — FastAPI prediction endpoint

Loads the Production model from MLflow registry and serves predictions.

Endpoints:
  GET  /              → health check
  GET  /model-info    → which model version is in production
  POST /predict       → predict PM2.5 from pollutant readings
  POST /predict/batch → predict for multiple stations at once

How it works:
- On startup, loads model from MLflow: models:/aqi-pm25-predictor@production
- Model is loaded ONCE and kept in memory — not reloaded per request
- If you promote a new model, restart the API to pick it up

Run locally:
  uvicorn api.main:app --reload --port 8000

Test it:
  curl http://localhost:8000/
  curl -X POST http://localhost:8000/predict \
       -H "Content-Type: application/json" \
       -d '{"CO": 45, "NH3": 7, "NO2": 30, "OZONE": 80, "PM10": 120, "SO2": 15}'
"""

import os
import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

DAGSHUB_USERNAME     = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_TOKEN        = os.getenv("DAGSHUB_TOKEN")
DAGSHUB_TRACKING_URI = os.getenv("DAGSHUB_TRACKING_URI")

if not DAGSHUB_TRACKING_URI:
    raise ValueError("DAGSHUB_TRACKING_URI not set in environment")

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
mlflow.set_tracking_uri(DAGSHUB_TRACKING_URI)

MODEL_URI = "models:/aqi-pm25-predictor@production"
FEATURES  = ["CO", "NH3", "NO2", "OZONE", "PM10", "SO2"]

# ── Load model once on startup ────────────────────────────────────────────────
print(f"Loading production model from MLflow: {MODEL_URI}")
model = mlflow.sklearn.load_model(MODEL_URI)
print("Model loaded successfully.")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AQI PM2.5 Predictor",
    description="Predicts PM2.5 levels for Delhi monitoring stations using pollutant readings.",
    version="1.0.0"
)


# ── Request / Response schemas ────────────────────────────────────────────────

class PollutantInput(BaseModel):
    CO:    Optional[float] = Field(None, description="Carbon Monoxide (µg/m³)")
    NH3:   Optional[float] = Field(None, description="Ammonia (µg/m³)")
    NO2:   Optional[float] = Field(None, description="Nitrogen Dioxide (µg/m³)")
    OZONE: Optional[float] = Field(None, description="Ozone (µg/m³)")
    PM10:  Optional[float] = Field(None, description="Particulate Matter 10 (µg/m³)")
    SO2:   Optional[float] = Field(None, description="Sulfur Dioxide (µg/m³)")

    class Config:
        json_schema_extra = {
            "example": {
                "CO": 45.0,
                "NH3": 7.0,
                "NO2": 30.0,
                "OZONE": 80.0,
                "PM10": 120.0,
                "SO2": 15.0
            }
        }


class PredictionResponse(BaseModel):
    predicted_pm25: float
    unit: str = "µg/m³"
    model: str = MODEL_URI
    note: str = "Null features are imputed with training mean"


class BatchInput(BaseModel):
    stations: List[PollutantInput]


class BatchResponse(BaseModel):
    predictions: List[float]
    unit: str = "µg/m³"
    count: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "AQI PM2.5 Predictor",
        "model": MODEL_URI
    }


@app.get("/model-info")
def model_info():
    """Returns which model version is currently loaded."""
    client = mlflow.tracking.MlflowClient()
    try:
        version = client.get_model_version_by_alias("aqi-pm25-predictor", "production")
        return {
            "model_name":    "aqi-pm25-predictor",
            "version":       version.version,
            "alias":         "production",
            "description":   version.description,
            "tracking_uri":  DAGSHUB_TRACKING_URI,
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not fetch model info: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PollutantInput):
    """
    Predict PM2.5 from pollutant readings for a single station.
    Missing features (null) are handled by the pipeline's imputer automatically.
    """
    try:
        input_df = pd.DataFrame([{
            "CO":    data.CO,
            "NH3":   data.NH3,
            "NO2":   data.NO2,
            "OZONE": data.OZONE,
            "PM10":  data.PM10,
            "SO2":   data.SO2,
        }])[FEATURES]

        prediction = model.predict(input_df)[0]

        return PredictionResponse(
            predicted_pm25=round(float(prediction), 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(data: BatchInput):
    """
    Predict PM2.5 for multiple stations in one request.
    """
    try:
        rows = []
        for station in data.stations:
            rows.append({
                "CO":    station.CO,
                "NH3":   station.NH3,
                "NO2":   station.NO2,
                "OZONE": station.OZONE,
                "PM10":  station.PM10,
                "SO2":   station.SO2,
            })

        input_df    = pd.DataFrame(rows)[FEATURES]
        predictions = model.predict(input_df)

        return BatchResponse(
            predictions=[round(float(p), 2) for p in predictions],
            count=len(predictions)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
