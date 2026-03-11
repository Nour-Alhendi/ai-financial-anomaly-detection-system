import pandas as pd
from pathlib import Path
from datetime import datetime


# =====================================================
# Schema Validation Module
# Validates structure, types, and content of asset
# time series data before any downstream processing.
# =====================================================

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
COLUMN_ORDER     = ["Date", "Open", "High", "Low", "Close", "Volume"]

# Alternative column names that get mapped to standard names
COLUMN_ALIASES = {
    "date"          : "Date",
    "timestamp"     : "Date",
    "time"          : "Date",
    "open"          : "Open",
    "high"          : "High",
    "lo"            : "Low",
    "low"           : "Low",
    "close"         : "Close",
    "closing_price" : "Close",
    "adj close"     : "Close",
    "adj_close"     : "Close",
    "vol"           : "Volume",
    "volume"        : "Volume",
}

# Columns to drop if present (not needed for analysis)
COLUMNS_TO_DROP = ["Adj Close", "Dividends", "Stock Splits", "Capital Gains"]

LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------
# Function: check_schema
# Purpose : Validate and normalize schema of a single
#           parquet file. Returns cleaned DataFrame
#           and a list of log messages.
# Input   : parquet file path
# Output  : (cleaned DataFrame or None, list of issues)
# -----------------------------------------------------
def check_schema(file_path):

    df       = pd.read_parquet(file_path)
    issues   = []
    changes  = []

    print(f"\nSchema check for {file_path.name}")

    # Empty dataset check
    if len(df) == 0:
        issues.append("Dataset is empty")
        print(f"  FAIL: Dataset is empty")
        return None, issues, changes

    # Duplicate column check
    # Same values → drop duplicate, same values → keep one, different values → reject file
    if df.columns.duplicated().any():
        dup_col_names = df.columns[df.columns.duplicated(keep=False)].unique().tolist()
        conflict = False

        for col in dup_col_names:
            col_versions = df.loc[:, df.columns == col]
            if col_versions.nunique(axis=1).max() > 1:
                # values differ → reject
                issues.append(f"Duplicate column '{col}' has conflicting values — manual resolution required")
                print(f"  FAIL: Duplicate column '{col}' with conflicting values")
                conflict = True
            else:
                # values identical → drop the duplicate silently
                df = df.loc[:, ~df.columns.duplicated(keep="first")]
                changes.append(f"Dropped identical duplicate column: '{col}'")
                print(f"  INFO: Dropped identical duplicate column: '{col}'")

        if conflict:
            return None, issues, changes

    # Drop unnecessary columns
    cols_dropped = [col for col in COLUMNS_TO_DROP if col in df.columns]
    if cols_dropped:
        df = df.drop(columns=cols_dropped)
        changes.append(f"Dropped columns: {cols_dropped}")
        print(f"  INFO: Dropped columns: {cols_dropped}")

    # Column name normalization
    # Map alternative names to standard names
    rename_map = {}
    for col in df.columns:
        normalized = COLUMN_ALIASES.get(col.strip().lower())
        if normalized and col != normalized:
            rename_map[col] = normalized

    if rename_map:
        df = df.rename(columns=rename_map)
        changes.append(f"Renamed columns: {rename_map}")
        print(f"  INFO: Renamed columns: {rename_map}")

    # Required columns check
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        print(f"  FAIL: Missing required columns: {missing_cols}")
        return None, issues, changes

    # Data type validation — Date: datetime, OHLC: float, Volume: int
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        try:
            df["Date"] = pd.to_datetime(df["Date"])
            changes.append("Converted Date to datetime")
            print(f"  INFO: Converted Date to datetime")
        except Exception:
            issues.append("Date column could not be converted to datetime")
            print(f"  FAIL: Date column could not be converted to datetime")

    for col in ["Open", "High", "Low", "Close"]:
        if not pd.api.types.is_float_dtype(df[col]):
            try:
                df[col] = df[col].astype(float)
                changes.append(f"Converted {col} to float")
                print(f"  INFO: Converted {col} to float")
            except Exception:
                issues.append(f"{col} could not be converted to float")
                print(f"  FAIL: {col} could not be converted to float")

    if not pd.api.types.is_integer_dtype(df["Volume"]):
        try:
            df["Volume"] = df["Volume"].astype(float).round(0).astype("Int64")
            changes.append("Converted Volume to integer")
            print(f"  INFO: Converted Volume to integer")
        except Exception:
            issues.append("Volume could not be converted to integer")
            print(f"  FAIL: Volume could not be converted to integer")

    # Column order standardization
    extra_cols = [col for col in df.columns if col not in COLUMN_ORDER]
    df = df[COLUMN_ORDER + extra_cols]

    # Summary
    if not issues:
        print(f"  OK — all checks passed")

    return df, issues, changes


# -----------------------------------------------------
# Function: run_schema_validation
# Purpose : Run schema validation for all asset files
#           and write a log file with results
# -----------------------------------------------------
def run_schema_validation():

    data_folder = Path("data/raw/raw_corrupted")
    files       = list(data_folder.glob("*.parquet"))

    if not files:
        print(f"No parquet files found in {data_folder}")
        return

    print(f"\nFound {len(files)} files. Running schema validation...\n")

    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_lines   = [f"Schema Validation Log — {timestamp}\n"]
    alert_lines = [f"Schema Alerts — {timestamp}\n"]
    passed      = 0

    for file in files:
        df, issues, changes = check_schema(file)

        log_lines.append(f"\n[{file.name}]")

        if changes:
            for change in changes:
                log_lines.append(f"  CHANGE : {change}")

        if issues:
            for issue in issues:
                log_lines.append(f"  FAIL   : {issue}")
            alert_lines.append(f"  [{file.name}] {' | '.join(issues)}")
        else:
            log_lines.append(f"  RESULT : OK")
            passed += 1

    # write full log
    log_path = LOG_DIR / "schema_validation.log"
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))

    # write alerts-only log
    alert_path = LOG_DIR / "schema_alerts.log"
    with open(alert_path, "w") as f:
        if len(alert_lines) == 1:
            f.write("\n".join(alert_lines) + "\n  No issues detected — all files passed")
        else:
            f.write("\n".join(alert_lines))

    print(f"\nSchema Validation Summary: {passed}/{len(files)} files passed")
    print(f"Full log : {log_path}")
    print(f"Alerts   : {alert_path}")


# -----------------------------------------------------
# Entry Point
# -----------------------------------------------------
if __name__ == "__main__":
    run_schema_validation()
