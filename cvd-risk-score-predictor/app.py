import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

MODEL_PATH = Path("model/pipeline.pkl")
DEFAULTS_PATH = Path("model/feature_defaults.json")
COLS_PATH = Path("model/feature_columns.json")

TARGET_NAME = "CVD Risk Score"

st.set_page_config(page_title="CVD Risk Score Predictor", layout="centered")

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    defaults = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))
    cols = json.loads(COLS_PATH.read_text(encoding="utf-8"))
    return model, defaults, cols

def make_row_from_defaults(defaults: dict, cols: list) -> dict:
    return {c: defaults.get(c, None) for c in cols}

def main():
    st.title("CVD Risk Score Predictor")
    st.caption("Enter patient information to estimate CVD Risk Score. This is a demo ML app (not medical advice).")

    if not MODEL_PATH.exists():
        st.error("Model not found. Please run train.py first (to generate model/pipeline.pkl).")
        st.stop()

    model, defaults, cols = load_assets()

    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])

    with tab1:
        st.subheader("Single Prediction")

        # Start with defaults for ALL required columns, then overwrite with user inputs for a few important ones
        row = make_row_from_defaults(defaults, cols)

        # --- Create a small, user-friendly form (key fields) ---
        # We only ask for common key fields; rest are filled using defaults computed from training data.
        # If a field doesn't exist in dataset columns, we skip it safely.
        def maybe_input_text(col_name, label=None):
            if col_name in row:
                row[col_name] = st.text_input(label or col_name, value=str(row[col_name] or ""))

        def maybe_input_number(col_name, label=None, step=1.0):
            if col_name in row:
                try:
                    default_val = float(row[col_name]) if row[col_name] is not None else 0.0
                except:
                    default_val = 0.0
                row[col_name] = st.number_input(label or col_name, value=default_val, step=step)

        left, right = st.columns(2)

        with left:
            maybe_input_text("Sex")
            maybe_input_number("Age", step=1.0)
            maybe_input_number("BMI", step=0.1)
            maybe_input_number("Systolic BP", step=1.0)
            maybe_input_number("Diastolic BP", step=1.0)

        with right:
            maybe_input_number("Total Cholesterol (mg/dL)", step=1.0)
            maybe_input_number("HDL (mg/dL)", step=1.0)
            maybe_input_number("Fasting Blood Sugar (mg/dL)", step=1.0)
            maybe_input_text("Smoking Status")
            maybe_input_text("Diabetes Status")

        # optional extra categorical fields (if present)
        maybe_input_text("Physical Activity Level")
        maybe_input_text("Family History of CVD")
        maybe_input_text("Blood Pressure Category")

        if st.button("Predict Risk Score"):
            X_input = pd.DataFrame([row], columns=cols)
            pred = float(model.predict(X_input)[0])
            st.success(f"Predicted {TARGET_NAME}: **{pred:.3f}**")

    with tab2:
        st.subheader("Batch Prediction (CSV)")
        st.write("Upload a CSV containing the same feature columns used in training.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded:
            df = pd.read_csv(uploaded)

            # Remove target/leak columns if user includes them
            for bad in ["CVD Risk Level", TARGET_NAME]:
                if bad in df.columns:
                    df = df.drop(columns=[bad])

            # Ensure all expected columns exist
            missing = [c for c in cols if c not in df.columns]
            for c in missing:
                df[c] = defaults.get(c, None)

            df = df[cols]
            preds = model.predict(df)
            out = df.copy()
            out[TARGET_NAME] = preds

            st.dataframe(out.head(20))
            st.download_button(
                "Download Predictions CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="cvd_predictions.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
