import requests
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()

API_KEY = os.getenv("DATA_GOV_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"

def fetch_aqi_data():
    url = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": 500,
        "filters[city]": "Delhi"
    }
    response = requests.get(url, params=params)

    print(f"API status code : {response.status_code}")
    print(f"API response    : {response.text[:500]}")

    if response.status_code != 200 or not response.text.strip():
        raise ValueError(f"API returned empty or error response. Status: {response.status_code}. Body: {response.text[:500]}")

    data = response.json()

    if "records" not in data:
        raise ValueError(f"API response missing 'records'. Response: {data}")

    records = data["records"]
    df = pd.DataFrame(records)
    df["fetched_at"] = datetime.now()

    # fix "NA" string → proper None so PostgreSQL accepts it
    for col in ["min_value", "max_value", "avg_value"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].where(df[col].notna(), other=None)

    return df
def save_to_db(df):
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS aqi_raw (
            id SERIAL PRIMARY KEY,
            country TEXT,
            state TEXT,
            city TEXT,
            station TEXT,
            pollutant_id TEXT,
            min_value FLOAT,
            max_value FLOAT,
            avg_value FLOAT,
            last_update TEXT,
            latitude TEXT,
            longitude TEXT,
            fetched_at TIMESTAMP
        )
    """)

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO aqi_raw 
            (country, state, city, station, pollutant_id, min_value, max_value, avg_value, last_update, latitude, longitude, fetched_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            row.get("country"),
            row.get("state"),
            row.get("city"),
            row.get("station"),
            row.get("pollutant_id"),
            row.get("min_value"),
            row.get("max_value"),
            row.get("avg_value"),
            row.get("last_update"),
            row.get("latitude"),
            row.get("longitude"),
            row.get("fetched_at")
        ))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Saved {len(df)} rows to Neon database")

if __name__ == "__main__":
    print("Fetching AQI data for Delhi...")
    df = fetch_aqi_data()
    print(f"Fetched {len(df)} rows")
    print(df.head())
    save_to_db(df)