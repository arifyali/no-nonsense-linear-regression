"""
Prepare CFPB complaint data for the linear regression guide.

Downloads instructions: see data/README.md
Usage: python data/prepare_data.py
"""

import os
import sys
import zipfile
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_FILE = os.path.join(DATA_DIR, "complaints.csv.zip")
RAW_FILE = os.path.join(DATA_DIR, "complaints.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "complaints_sample.csv")
SAMPLE_SIZE = 50_000
RANDOM_SEED = 42


def main():
    if os.path.exists(ZIP_FILE):
        print(f"Reading from zip: {ZIP_FILE}")
        with zipfile.ZipFile(ZIP_FILE) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            print(f"  Extracting: {csv_name}")
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, low_memory=False)
    elif os.path.exists(RAW_FILE):
        print(f"Reading from CSV: {RAW_FILE}")
        df = pd.read_csv(RAW_FILE, low_memory=False)
    else:
        print(f"Data not found. Looked for:")
        print(f"  {ZIP_FILE}")
        print(f"  {RAW_FILE}")
        print()
        print("Download it first:")
        print("  1. Go to https://www.consumerfinance.gov/data-research/consumer-complaints/")
        print("  2. Click 'Download the data' → CSV format")
        print(f"  3. Save the zip in {DATA_DIR}/")
        sys.exit(1)
    print(f"  Raw records: {len(df):,}")

    # Keep only rows with actual complaint text
    df = df.dropna(subset=["Consumer complaint narrative"])
    print(f"  With narratives: {len(df):,}")

    # Parse dates and compute response time
    df["Date received"] = pd.to_datetime(df["Date received"], format="mixed")
    df["Date sent to company"] = pd.to_datetime(df["Date sent to company"], format="mixed")
    df["response_time_days"] = (df["Date sent to company"] - df["Date received"]).dt.days

    # Drop rows where response time is missing or negative
    df = df[df["response_time_days"] >= 0]
    print(f"  With valid response time: {len(df):,}")

    # Keep useful columns
    columns_to_keep = [
        "Date received",
        "Product",
        "Sub-product",
        "Issue",
        "Consumer complaint narrative",
        "Company",
        "State",
        "Company response to consumer",
        "Timely response?",
        "Consumer disputed?",
        "Date sent to company",
        "response_time_days",
    ]
    df = df[[c for c in columns_to_keep if c in df.columns]]

    # Sample
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
        print(f"  Sampled: {SAMPLE_SIZE:,} records")
    else:
        print(f"  Using all {len(df):,} records (below sample threshold)")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {OUTPUT_FILE}")
    print(f"File size: {os.path.getsize(OUTPUT_FILE) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
