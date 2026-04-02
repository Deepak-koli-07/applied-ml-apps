import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def load_from_db():
    conn = psycopg2.connect(DATABASE_URL)
    df = pd.read_sql("SELECT * FROM aqi_raw", conn)
    conn.close()
    return df

def preprocess(df):
    # keep only needed columns
    df = df[["station", "pollutant_id", "avg_value", "last_update"]]

    # drop rows where avg_value is null
    df = df.dropna(subset=["avg_value"])

    # pivot long → wide
    df_wide = df.pivot_table(
        index=["station", "last_update"],
        columns="pollutant_id",
        values="avg_value"
    ).reset_index()

    # flatten column names
    df_wide.columns.name = None

    # drop rows where all pollutant values are null
    pollutant_cols = [c for c in df_wide.columns if c not in ["station", "last_update"]]
    df_wide = df_wide.dropna(subset=pollutant_cols, how="all")

    # add processed timestamp
    df_wide["processed_at"] = datetime.now()

    return df_wide

def save_processed_to_db(df):
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    # get pollutant columns dynamically
    pollutant_cols = [c for c in df.columns if c not in ["station", "last_update", "processed_at"]]

    # build CREATE TABLE dynamically
    pollutant_col_defs = "\n".join([f'"{col}" FLOAT,' for col in pollutant_cols])

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS aqi_processed (
            id SERIAL PRIMARY KEY,
            station TEXT,
            last_update TEXT,
            {pollutant_col_defs}
            processed_at TIMESTAMP
        )
    """)

    # insert rows
    for _, row in df.iterrows():
        cols = ["station", "last_update"] + pollutant_cols + ["processed_at"]
        values = [row.get(c) for c in cols]
        placeholders = ", ".join(["%s"] * len(cols))
        col_names = ", ".join([f'"{c}"' for c in cols])

        cursor.execute(f"""
            INSERT INTO aqi_processed ({col_names})
            VALUES ({placeholders})
        """, values)

    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Saved {len(df)} processed rows to Neon")

if __name__ == "__main__":
    print("Loading raw data from Neon...")
    df = load_from_db()
    print(f"Loaded {len(df)} rows")

    print("Preprocessing...")
    df_processed = preprocess(df)
    print(f"After pivot: {df_processed.shape}")
    print(df_processed.head())

    save_processed_to_db(df_processed)