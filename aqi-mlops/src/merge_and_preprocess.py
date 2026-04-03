"""
merge_and_preprocess.py

Merges aqi_historical (90-day simulated) + aqi_raw (real live data)
in correct ascending date order, then preprocesses into wide format
and saves to aqi_processed_full table.

Why a new table (aqi_processed_full)?
- aqi_processed has only 43 rows (one real snapshot)
- aqi_processed_full has the full merged + pivoted dataset ready for ML
- We never overwrite real data tables

Order: historical first (oldest) → real data last (newest)
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from datetime import datetime
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set in environment")


def load_historical(conn):
    """Load 90-day simulated data from aqi_historical."""
    df = pd.read_sql("""
        SELECT station, pollutant_id, avg_value, last_update
        FROM aqi_historical
        WHERE city = 'Delhi'
        ORDER BY last_update ASC
    """, conn)
    df["source"] = "historical"
    print(f"Loaded {len(df)} rows from aqi_historical")
    return df


def load_real(conn):
    """Load real live data from aqi_raw."""
    df = pd.read_sql("""
        SELECT station, pollutant_id, avg_value, last_update
        FROM aqi_raw
        WHERE city = 'Delhi'
        ORDER BY last_update ASC
    """, conn)
    df["source"] = "real"
    print(f"Loaded {len(df)} rows from aqi_raw")
    return df


def merge_in_order(df_hist, df_real):
    """
    Concatenate historical + real in ascending date order.
    Historical comes first (older dates), real comes last (newest date).
    """
    df_merged = pd.concat([df_hist, df_real], ignore_index=True)

    # Sort by last_update ascending — oldest first
    # Parse the date strings for correct sorting
    df_merged["last_update_dt"] = pd.to_datetime(
        df_merged["last_update"], dayfirst=True, errors="coerce"
    )
    df_merged = df_merged.sort_values("last_update_dt").reset_index(drop=True)

    print(f"\nMerged dataset:")
    print(f"  Total rows     : {len(df_merged)}")
    print(f"  Earliest date  : {df_merged['last_update_dt'].min()}")
    print(f"  Latest date    : {df_merged['last_update_dt'].max()}")
    print(f"  Historical rows: {(df_merged['source'] == 'historical').sum()}")
    print(f"  Real rows      : {(df_merged['source'] == 'real').sum()}")

    return df_merged


def preprocess(df):
    """
    Pivot long format -> wide format.
    One row per (station, last_update) with each pollutant as a column.
    Same logic as preprocess.py but on the merged dataset.
    """
    df = df.dropna(subset=["avg_value"])

    df_wide = df.pivot_table(
        index=["station", "last_update", "last_update_dt", "source"],
        columns="pollutant_id",
        values="avg_value",
        aggfunc="mean"
    ).reset_index()

    df_wide.columns.name = None

    # Drop rows where all pollutant values are null
    pollutant_cols = [c for c in df_wide.columns
                      if c not in ["station", "last_update", "last_update_dt", "source"]]
    df_wide = df_wide.dropna(subset=pollutant_cols, how="all")

    # Sort final output ascending by date
    df_wide = df_wide.sort_values("last_update_dt").reset_index(drop=True)

    df_wide["processed_at"] = datetime.now()

    print(f"\nAfter pivot (wide format):")
    print(f"  Shape          : {df_wide.shape}")
    print(f"  Columns        : {df_wide.columns.tolist()}")
    print(f"  Earliest       : {df_wide['last_update_dt'].min()}")
    print(f"  Latest         : {df_wide['last_update_dt'].max()}")
    print(f"  Unique stations: {df_wide['station'].nunique()}")
    print(f"  Null counts    :")
    print(df_wide[pollutant_cols].isnull().sum().to_string())

    return df_wide, pollutant_cols


def save_to_db(df, pollutant_cols, conn):
    """Save wide-format merged data to aqi_processed_full."""
    cursor = conn.cursor()

    pollutant_col_defs = "\n".join([f'"{col}" FLOAT,' for col in pollutant_cols])

    cursor.execute(f"""
        DROP TABLE IF EXISTS aqi_processed_full;
        CREATE TABLE aqi_processed_full (
            id             SERIAL PRIMARY KEY,
            station        TEXT,
            last_update    TEXT,
            source         TEXT,
            {pollutant_col_defs}
            processed_at   TIMESTAMP
        )
    """)

    cols = ["station", "last_update", "source"] + pollutant_cols + ["processed_at"]

    rows = []
    for _, row in df.iterrows():
        rows.append(tuple(
            None if pd.isna(row.get(c)) else row.get(c)
            for c in cols
        ))

    col_names = ", ".join([f'"{c}"' for c in cols])
    insert_sql = f'INSERT INTO aqi_processed_full ({col_names}) VALUES %s'

    execute_values(cursor, insert_sql, rows, page_size=500)
    conn.commit()
    cursor.close()

    print(f"\nSaved {len(rows)} rows to aqi_processed_full")


def verify(conn):
    df = pd.read_sql("""
        SELECT
            source,
            COUNT(*)             AS rows,
            MIN(last_update)     AS earliest,
            MAX(last_update)     AS latest,
            COUNT(DISTINCT station) AS stations
        FROM aqi_processed_full
        GROUP BY source
        ORDER BY earliest ASC
    """, conn)
    print("\nVerification:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    print("Connecting to Neon DB...")
    with psycopg2.connect(DATABASE_URL) as conn:

        print("\nStep 1: Loading data...")
        df_hist = load_historical(conn)
        df_real = load_real(conn)

        print("\nStep 2: Merging in ascending date order...")
        df_merged = merge_in_order(df_hist, df_real)

        print("\nStep 3: Preprocessing (pivot long -> wide)...")
        df_wide, pollutant_cols = preprocess(df_merged)

        print("\nStep 4: Saving to aqi_processed_full...")
        save_to_db(df_wide, pollutant_cols, conn)

        print("\nStep 5: Verifying...")
        verify(conn)
