import pandas as pd
from pathlib import Path


# =====================================================
# Duplicate Check Module
# Detect duplicate trading dates in asset time series
# =====================================================


# -----------------------------------------------------
# Function: check_duplicates
# Purpose : Detect duplicate trading dates in a file
# Input   : parquet file path
# Output  : dataframe containing duplicate rows
# -----------------------------------------------------
def check_duplicates(file_path):

    # read asset data
    df = pd.read_parquet(file_path)

    # detect duplicate trading dates
    duplicate_rows = df[df.duplicated(subset=["Date"])]

    print(f"\nDuplicate check for {file_path.name}")

    if duplicate_rows.empty:
        print("No duplicate dates found")
    else:
        print("Duplicate rows detected:")
        print(duplicate_rows)

    return duplicate_rows


# -----------------------------------------------------
# Function: run_duplicate_check
# Purpose : Run duplicate checks for all asset files
# -----------------------------------------------------
def run_duplicate_check():

    data_folder = Path("data/raw/raw_corrupted")

    for file in data_folder.glob("*.parquet"):

        duplicates = check_duplicates(file)

        if duplicates.empty:
            print(f"{file.name} → OK")


# -----------------------------------------------------
# Entry Point
# -----------------------------------------------------
if __name__ == "__main__":
    run_duplicate_check()