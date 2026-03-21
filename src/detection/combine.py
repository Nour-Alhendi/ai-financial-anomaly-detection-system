# Combines all anomaly signals into a single score and combined_anomaly flag.
# Signals: z_anomaly (20d), z_anomaly_60 (60d), if_anomaly, ae_anomaly
# Columns: anomaly_score, combined_anomaly

import pandas as pd
from pathlib import Path

INPUT_DIR = Path("data/features")
OUTPUT_DIR = Path("data/detection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def combine(file_path):
    df = pd.read_parquet(file_path)
    df["anomaly_score"] = (
        df["z_anomaly"].fillna(False).astype(int) +
        df["z_anomaly_60"].fillna(False).astype(int) +
        df["if_anomaly"].fillna(False).astype(int) +
        df["ae_anomaly"].fillna(False).astype(int)
    )
    df["combined_anomaly"] = df["anomaly_score"] > 0
    return df


def run_combine():
    for file in INPUT_DIR.glob("*.parquet"):
        if file.stem == "^SPX":
            continue
        df = combine(file)
        df.to_parquet(OUTPUT_DIR / file.name)
        print(f"Saved: {file.name}")


if __name__ == "__main__":
    run_combine()
