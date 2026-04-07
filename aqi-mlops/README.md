# AQI MLOps Pipeline 🌫️

## Why This Project Was Created

Delhi consistently ranks among the most polluted cities in the world. Air Quality Index (AQI) data is published by the Central Pollution Control Board (CPCB) through the Government of India's open data platform — updated every hour, freely accessible.

Most ML projects use static datasets downloaded once from Kaggle. This project is different — it uses **live, daily-updating Indian government data** to build a production-style MLOps pipeline that automatically retrains, tracks, and serves models without any manual intervention.

The goal was to build something real — not a tutorial project, but an end-to-end system that mirrors how ML actually works in production companies.

---

## What This Project Does

1. **Fetches** real-time AQI data for Delhi from data.gov.in API every day
2. **Stores** raw data in a cloud PostgreSQL database (Neon)
3. **Preprocesses** data — cleans nulls, pivots long format to wide format
4. **Trains** multiple ML models to predict PM2.5 levels from other pollutants
5. **Tracks** all experiments — parameters, metrics, artifacts — via MLflow on Dagshub
6. **Registers** the best model in MLflow Model Registry
7. **Promotes** model from Staging to Production automatically if it beats the current model
8. **Serves** predictions via a FastAPI REST endpoint
9. **Automates** the entire pipeline daily using GitHub Actions

---

## What I Learned

### Data Engineering
- Fetching data from a real government REST API
- Handling messy real-world data (NA strings, missing sensors, long format)
- Storing and querying data in cloud PostgreSQL (Neon)
- Long to wide format pivoting for ML feature preparation

### MLflow
- Experiment tracking — logging parameters, metrics, and artifacts
- Comparing multiple model runs in MLflow UI
- Model Registry — registering, versioning, and staging models
- Promoting models from Staging to Production programmatically
- Hosting MLflow tracking server on Dagshub (free)

### MLOps & Automation
- GitHub Actions — writing YAML workflows for CI/CD
- Automated daily retraining triggered by cron schedule
- Passing secrets securely to GitHub Actions via GitHub Secrets
- Understanding why fresh machines need external storage (Neon, Dagshub)

### Model Serving
- Loading a registered MLflow model by stage name
- Wrapping it in a FastAPI endpoint
- Separating ML logic from API logic cleanly

### Git & GitHub
- Feature branch workflow
- Committing only relevant files
- Using .gitignore properly
- Connecting GitHub repo to Dagshub for ML tracking

---

## What I Achieved

- Built a fully automated MLOps pipeline on real Indian government data
- Zero manual steps after setup — data flows from API to prediction endpoint automatically
- Learned MLflow end to end — tracking, registry, serving — through a real project
- Understood GitHub Actions not from docs but by actually running pipelines
- Created a portfolio project that is genuinely different from standard Kaggle projects

---

## Tech Stack

| Tool | Purpose |
|---|---|
| data.gov.in API | Live AQI data source |
| Python | Core language |
| Pandas | Data cleaning and transformation |
| Scikit-learn / XGBoost | ML models |
| MLflow | Experiment tracking and model registry |
| Dagshub | Free hosted MLflow server and artifact storage |
| Neon PostgreSQL | Cloud database for raw and processed data |
| FastAPI | Model serving REST API |
| GitHub Actions | Pipeline automation and scheduling |
| Docker | Containerizing FastAPI for deployment |

---

## Project Structure

```
aqi-mlops/
├── data/                          # raw data files (not committed)
├── notebooks/
│   └── explore.ipynb              # data exploration and model validation
├── src/
│   ├── fetch_data.py              # fetch AQI from API → save to Neon
│   ├── preprocess.py              # clean and pivot data → save to Neon
│   ├── train.py                   # train models → log to MLflow
│   ├── evaluate.py                # compare new model vs production
│   └── promote.py                 # promote best model in registry
├── api/
│   └── main.py                    # FastAPI serving MLflow model
├── .github/
│   └── workflows/
│       ├── retrain.yml            # daily automated retraining pipeline
│       └── deploy.yml             # deploy on model promotion
├── requirements.txt
├── .env                           # local secrets (never committed)
└── README.md
```

---

## ML Problem

**Task:** Regression — Predict PM2.5 levels at Delhi monitoring stations

**Features:** CO, NH3, NO2, OZONE, PM10, SO2

**Target:** PM2.5

**Why PM2.5:** It is the most health-critical pollutant, used as the primary indicator by CPCB and WHO. Predicting it from other pollutants is useful when PM2.5 sensors malfunction or are unavailable at a station.

---

## Data Source

- **API:** [data.gov.in](https://data.gov.in) — Real Time Air Quality Index from Various Locations
- **Published by:** Central Pollution Control Board (CPCB), Ministry of Environment, Forest and Climate Change, Government of India
- **Update frequency:** Hourly
- **Coverage:** 285+ monitoring stations across Delhi

---

*Built as part of an MLOps learning journey — transitioning from Data Analyst to AI/ML Engineer.*
