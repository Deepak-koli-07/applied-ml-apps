import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


DATA_PATH = "CVD Dataset.csv"  # put the CSV here for local training, or change path
TARGET = "CVD Risk Score"
LEAK_COLS = ["CVD Risk Level"]  # don't use this as a feature


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_features = X.select_dtypes(include="number").columns
    cat_features = X.select_dtypes(include="object").columns

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ])


def compute_feature_defaults(X: pd.DataFrame) -> dict:
    """
    Defaults used by the app if a user doesn't provide some fields.
    Numeric -> median
    Categorical -> mode
    """
    defaults = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            defaults[col] = float(X[col].median()) if X[col].notna().any() else 0.0
        else:
            # mode might be empty if all NaN
            mode_vals = X[col].mode(dropna=True)
            defaults[col] = str(mode_vals.iloc[0]) if len(mode_vals) else ""
    return defaults


def main():
    df = pd.read_csv(DATA_PATH)

    # basic cleanup: drop leakage / target-like columns from features
    drop_cols = [TARGET] + [c for c in LEAK_COLS if c in df.columns]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # target: fill missing with median (same spirit as your notebook)
    y = df[TARGET]
    y = y.fillna(y.median())

    preprocessor = build_preprocessor(X)

    model = Pipeline([
        ("prep", preprocessor),
        ("lin", LinearRegression()),
    ])

    model.fit(X, y)

    out_dir = Path("model")
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / "pipeline.pkl")

    defaults = compute_feature_defaults(X)
    with open(out_dir / "feature_defaults.json", "w", encoding="utf-8") as f:
        json.dump(defaults, f, indent=2)

    with open(out_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2)

    print("Saved:")
    print(" - model/pipeline.pkl")
    print(" - model/feature_defaults.json")
    print(" - model/feature_columns.json")


if __name__ == "__main__":
    main()
