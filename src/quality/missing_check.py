import pandas as pd
from pathlib import Path

def check_missing_values(file_path):
    """
    Check missing values in a parquet file
    """

    df = pd.read_parquet(file_path)

    missing_report = df.isna().sum()

    print(f"\nMissing values report for {file_path.name}")
    print(missing_report)

    return missing_report


def run_missing_check():
    """
    Run missing value checks for all assets
    """

    data_folder = Path("data/raw/raw_corrupted")

    for file in data_folder.glob("*.parquet"):
        missing_report = check_missing_values(file)

        total_missing = missing_report.sum()

        if total_missing == 0:
            print(f"{file.name} → No missing values")
        else:
            print(f"{file.name} → Missing values detected:")
            print(missing_report)


if __name__ == "__main__":
    run_missing_check()