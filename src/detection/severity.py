# Classifies anomaly severity based on how many models flagged the row.
# 4 signals: z_anomaly, z_anomaly_60, if_anomaly, ae_anomaly
# Columns: severity

import pandas as pd
from pathlib import Path

INPUT_DIR = Path("data/detection")
OUTPUT_DIR = Path("data/detection")


def classify_severity(score):
    if score == 0:
        return "normal"
    if score == 1:
        return "watch"
    if score == 2:
        return "warning"
    return "critical"


def run_severity():
    for file in INPUT_DIR.glob("*.parquet"):
        if file.stem == "^SPX":
            continue
        df = pd.read_parquet(file)
        df["severity"] = df["anomaly_score"].apply(classify_severity)
        df.to_parquet(OUTPUT_DIR / file.name)
        print(f"Saved: {file.name}")


if __name__ == "__main__":
    run_severity()
