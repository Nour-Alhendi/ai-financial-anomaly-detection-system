# =====================================================
# Gap Check Module
# Detect missing trading days in asset time series
# =====================================================

import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Funktion: check_time_gaps
# Purpose: Detect gaps in trading dates
# --------------------------------------------------
def check_time_gaps(file_path):
    df = pd.read_parquet(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df["gap"] = df["Date"].diff()
    gaps = df[df["gap"] > pd.Timedelta(days=3)]
    print(f"\nGap check for {file_path.name}")
    if gaps.empty:
        print("No suspicious gaps found")
    else:
        print("Potential gaps detected:")
        print(gaps[["Date", "gap"]])
    return gaps

# =====================================================
# Function: run_gap_check
# =====================================================
def run_gap_check():
    data_folder = Path("data/raw/raw_corrupted")
    for file in data_folder.glob("*.parquet"):
        gaps = check_time_gaps(file)
        
        if gaps.empty:
            print(f"{file.name} -> OK")


# -----------------------------------------------------
# Entry Point
# -----------------------------------------------------
if __name__ == "__main__":
    run_gap_check()