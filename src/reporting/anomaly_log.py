"""
FinWatch AI — Layer 8A: Anomaly Log
=====================================
Appends each pipeline run's decisions to a persistent audit log.

Output: data/logs/anomaly_log.parquet
  Columns: run_id, timestamp, ticker, date, severity, action, confidence, context
"""

import uuid
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "data/logs"
LOG_PATH = LOG_DIR / "anomaly_log.parquet"


def log(decisions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append decisions from one pipeline run to the audit log.

    Args:
        decisions_df: DataFrame from decisions.parquet
                      (must have: ticker, date, severity, action, confidence, context)

    Returns:
        Full log DataFrame (all historical runs).
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    run_id    = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()

    entry = decisions_df[["ticker", "date", "severity", "action", "confidence", "context"]].copy()
    entry.insert(0, "run_id",    run_id)
    entry.insert(1, "timestamp", timestamp)

    if LOG_PATH.exists():
        existing = pd.read_parquet(LOG_PATH)
        combined = pd.concat([existing, entry], ignore_index=True)
    else:
        combined = entry

    combined.to_parquet(LOG_PATH, index=False)
    print(f"[anomaly_log] Run {run_id} — logged {len(entry)} decisions → {LOG_PATH}")
    return combined


if __name__ == "__main__":
    decisions_path = ROOT / "data/decisions/decisions.parquet"
    df = pd.read_parquet(decisions_path)
    log(df)
