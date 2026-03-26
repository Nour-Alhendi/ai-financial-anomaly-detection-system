"""
FinWatch AI — Layer 6: Decision Pipeline
=========================================
Loads outputs from Layer 4 (detection) and Layer 5 (prediction),
merges them per ticker, runs the Decision Engine, and saves results.

Output: data/decisions/decisions.parquet
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from prediction.models.xgboost_risk      import load_data, predict as predict_risk
from prediction.models.xgboost_direction import predict as predict_direction
from decision.decision_engine     import run_decision_engine

DATA_DIR = ROOT / "data/detection"
OUT_DIR  = ROOT / "data/decisions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fields needed from detection parquets (latest row per ticker)
DETECTION_FIELDS = [
    "ticker", "Date",
    "anomaly_score", "market_anomaly", "sector_anomaly",
    "es_ratio",
    "rsi", "momentum_5", "momentum_10",
    "max_drawdown_30d", "obv_signal",
    "excess_return",
]


def _load_latest_detection(cutoff_date=None) -> pd.DataFrame:
    """Load the most recent row per ticker from detection parquets.

    Args:
        cutoff_date: if provided, only rows on or before this date are considered.
                     Useful for backtesting on historical snapshots.
    """
    rows = []
    for f in sorted(DATA_DIR.glob("*.parquet")):
        if f.stem.startswith("^"):
            continue
        df = pd.read_parquet(f)
        df["ticker"] = f.stem
        df["Date"] = pd.to_datetime(df["Date"])

        if cutoff_date is not None:
            df = df[df["Date"] <= cutoff_date]
            if df.empty:
                print(f"Skipping {f.stem} — no data on or before {cutoff_date.date()}")
                continue

        missing = [c for c in DETECTION_FIELDS if c not in df.columns]
        if missing:
            print(f"Skipping {f.stem} — missing columns: {missing}")
            continue

        latest = df.sort_values("Date").iloc[-1]
        rows.append(latest[DETECTION_FIELDS])

    return pd.DataFrame(rows).reset_index(drop=True)


def run(cutoff_date=None):
    print("=" * 55)
    print("DECISION PIPELINE — Layer 6")
    print("=" * 55)

    cutoff = pd.Timestamp(cutoff_date) if cutoff_date else None
    if cutoff:
        print(f"  Snapshot date: {cutoff.date()}")

    # ── 1. Load shared data (used by both prediction models)
    print("\n[1/4] Loading detection data...")
    data = load_data()

    # ── 2. Get latest predictions from Layer 5
    print("[2/4] Running Risk + Direction predictions...")
    risk_df      = predict_risk(data)       # ticker, risk_level, p_low, p_high, anomaly_score
    direction_df = predict_direction(data)  # ticker, direction, p_up, p_stable, p_down

    # ── 3. Load latest detection signals (anomaly, ES, momentum, drawdown)
    print("[3/4] Loading latest detection signals...")
    detection_df = _load_latest_detection(cutoff)

    # ── 4. Merge all inputs on ticker
    merged = (
        risk_df[["ticker", "risk_level", "p_high"]]
        .merge(direction_df[["ticker", "direction", "p_up", "p_down"]], on="ticker")
        .merge(detection_df, on="ticker")
    )

    # ── 5. Build records for Decision Engine
    records = []
    for _, row in merged.iterrows():
        records.append({
            "ticker":         row["ticker"],
            "date":           str(row["Date"].date()),
            "risk_level":     row["risk_level"],
            "p_high":         float(row["p_high"]),
            "direction":      row["direction"],
            "p_down":         float(row["p_down"]),
            "p_up":           float(row["p_up"]),
            "anomaly_score":  int(row["anomaly_score"]),
            "market_anomaly": bool(row["market_anomaly"]),
            "sector_anomaly": bool(row["sector_anomaly"]),
            "es_ratio":       float(row["es_ratio"])       if pd.notna(row["es_ratio"])       else 0.0,
            "rsi":            float(row["rsi"])             if pd.notna(row["rsi"])             else 50.0,
            "momentum_5":     float(row["momentum_5"])     if pd.notna(row["momentum_5"])     else 0.0,
            "momentum_10":    float(row["momentum_10"])    if pd.notna(row["momentum_10"])    else 0.0,
            "drawdown":       float(row["max_drawdown_30d"]) if pd.notna(row["max_drawdown_30d"]) else 0.0,
            "excess_return":  float(row["excess_return"])    if pd.notna(row["excess_return"])    else 0.0,
            "obv_signal":     float(row["obv_signal"])       if pd.notna(row["obv_signal"])       else 0.0,
        })

    # ── 6. Run Decision Engine
    print(f"[4/4] Running Decision Engine on {len(records)} tickers...")
    decisions = run_decision_engine(records)

    # ── 7. Save to parquet
    out_df = pd.DataFrame([vars(d) for d in decisions])
    out_path = OUT_DIR / "decisions.parquet"
    out_df.to_parquet(out_path, index=False)

    # ── 8. Print summary
    print(f"\nSaved: {out_path}")
    print(f"\n{'Ticker':<8} {'Severity':<20} {'Action':<10} {'Conf':>5}  Context")
    print("-" * 65)
    for d in sorted(decisions, key=lambda x: x.severity):
        print(f"{d.ticker:<8} {d.severity:<20} {d.action:<10} {d.confidence:>4.0%}  {d.context}")

    print("\n" + "=" * 55)
    print("Decision Pipeline complete.")
    print("=" * 55)

    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Snapshot date for backtesting, e.g. 2023-07-01",
    )
    args = parser.parse_args()
    run(cutoff_date=args.date)
