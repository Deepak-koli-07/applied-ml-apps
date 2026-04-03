"""
backfill_historical.py

Generates 90 days of simulated historical AQI data for Delhi stations
and stores it in a separate `aqi_historical` table in Neon DB.

Why a separate table?
- `aqi_raw` contains real data fetched from data.gov.in API
- We never mix simulated data with real data
- `aqi_historical` is clearly labelled as backfill/simulated

Schema matches `aqi_raw` exactly so the same preprocess.py can work on both.
Data is inserted in ascending date order (oldest first).
Duplicate rows are skipped safely via ON CONFLICT DO NOTHING.
Only Delhi stations are included.
"""

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from datetime import datetime, timedelta
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set in environment")

# ── Pollutants that match what data.gov.in API returns ────────────────────────
POLLUTANTS = ["CO", "NH3", "NO2", "OZONE", "PM10", "PM2.5", "SO2"]

# Stations that have no NH3/SO2 readings (matches real data)
NO_NH3_SO2 = {
    "Burari Crossing, Delhi - IMD",
    "CRRI Mathura Road, Delhi - IMD",
    "IGI Airport (T3), Delhi - IMD",
    "Mundka, Delhi - IMD",
    "Narela, Delhi - IMD",
    "Pusa, Delhi - IMD",
    "Rohini, Delhi - IMD",
}


def load_station_baselines(conn):
    """Load real station snapshots from aqi_processed as base values."""
    df = pd.read_sql(
        "SELECT * FROM aqi_processed WHERE station LIKE '%Delhi%'",
        conn
    )
    print(f"Loaded {len(df)} Delhi station baselines from aqi_processed")
    return df


def generate_historical_long(df_snap):
    """
    Generate 90-day historical data in LONG format matching aqi_raw schema.

    Long format = one row per (station, pollutant, timestamp)
    Same structure as what fetch_data.py inserts from the API.
    """
    np.random.seed(42)

    end_date   = datetime(2026, 4, 2, 16, 0, 0)   # matches our latest real snapshot
    start_date = end_date - timedelta(days=90)

    # Every 4 hours — 6 readings per day per station
    timestamps = pd.date_range(start=start_date, end=end_date, freq="4h")

    records = []

    # Ascending order: oldest timestamp first
    for ts in sorted(timestamps):
        hour = ts.hour

        # Real-world pattern: peaks at morning rush (8am) and evening (10pm)
        time_factor = 1.0 + 0.25 * np.sin((hour - 8) * np.pi / 12)

        for _, station_row in df_snap.iterrows():
            station = station_row["station"]

            # Only Delhi stations
            if "Delhi" not in station:
                continue

            day_offset = (ts - start_date).days
            # Seasonal: Delhi winters more polluted, April cleaner (~25% drop over 90d)
            seasonal = 1.0 - 0.003 * day_offset

            for pollutant in POLLUTANTS:
                base = station_row.get(pollutant, np.nan)

                # Skip pollutants this station doesn't report
                if pd.isna(base):
                    continue

                # Skip NH3/SO2 for stations that don't report them
                if pollutant in ("NH3", "SO2") and station in NO_NH3_SO2:
                    continue

                noise     = np.random.normal(1.0, 0.12)   # ±12% variation
                avg_value = round(max(0, base * seasonal * time_factor * noise), 1)
                min_value = round(avg_value * np.random.uniform(0.75, 0.95), 1)
                max_value = round(avg_value * np.random.uniform(1.05, 1.25), 1)

                records.append((
                    "India",                          # country
                    "Delhi",                          # state
                    "Delhi",                          # city
                    station,                          # station
                    pollutant,                        # pollutant_id
                    float(min_value),                 # min_value
                    float(max_value),                 # max_value
                    float(avg_value),                 # avg_value
                    ts.strftime("%d-%m-%Y %H:%M:%S"), # last_update (matches API format)
                    None,                             # latitude
                    None,                             # longitude
                    datetime.now(),                   # fetched_at
                ))

    print(f"Generated {len(records)} rows in long format (ascending date order)")
    return records


def create_table_if_not_exists(cursor):
    """Create aqi_historical with same schema as aqi_raw + unique constraint."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS aqi_historical (
            id           SERIAL PRIMARY KEY,
            country      TEXT,
            state        TEXT,
            city         TEXT,
            station      TEXT,
            pollutant_id TEXT,
            min_value    FLOAT,
            max_value    FLOAT,
            avg_value    FLOAT,
            last_update  TEXT,
            latitude     TEXT,
            longitude    TEXT,
            fetched_at   TIMESTAMP,
            UNIQUE (station, pollutant_id, last_update)
        )
    """)


def save_to_db(records, conn):
    """
    Insert records in ascending date order.
    ON CONFLICT DO NOTHING — safe to run multiple times, no duplicates.
    """
    cursor = conn.cursor()
    create_table_if_not_exists(cursor)

    insert_sql = """
        INSERT INTO aqi_historical
            (country, state, city, station, pollutant_id,
             min_value, max_value, avg_value, last_update,
             latitude, longitude, fetched_at)
        VALUES %s
        ON CONFLICT (station, pollutant_id, last_update) DO NOTHING
    """

    # Batch insert for speed
    batch_size = 1000
    inserted   = 0
    skipped    = 0

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        cursor.execute("SELECT COUNT(*) FROM aqi_historical")
        before = cursor.fetchone()[0]

        execute_values(cursor, insert_sql, batch)
        conn.commit()

        cursor.execute("SELECT COUNT(*) FROM aqi_historical")
        after = cursor.fetchone()[0]

        inserted += (after - before)
        skipped  += len(batch) - (after - before)

        print(f"  Batch {i // batch_size + 1}: inserted {after - before}, skipped {len(batch) - (after - before)}")

    cursor.close()
    print(f"\nDone. Total inserted: {inserted} | Skipped (duplicates): {skipped}")


def verify(conn):
    """Quick sanity check after insert."""
    df = pd.read_sql("""
        SELECT
            MIN(last_update)  AS earliest,
            MAX(last_update)  AS latest,
            COUNT(*)          AS total_rows,
            COUNT(DISTINCT station)      AS stations,
            COUNT(DISTINCT pollutant_id) AS pollutants
        FROM aqi_historical
        WHERE city = 'Delhi'
    """, conn)
    print("\n-- Verification --")
    print(df.to_string(index=False))


if __name__ == "__main__":
    print("Connecting to Neon DB...")
    with psycopg2.connect(DATABASE_URL) as conn:

        print("\nStep 1: Loading real station baselines...")
        df_snap = load_station_baselines(conn)

        print("\nStep 2: Generating 90-day historical data...")
        records = generate_historical_long(df_snap)

        print("\nStep 3: Saving to aqi_historical table (ascending order)...")
        save_to_db(records, conn)

        print("\nStep 4: Verifying...")
        verify(conn)
