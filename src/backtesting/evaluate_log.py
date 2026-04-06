"""
FinWatch AI — Decision Log Evaluator
======================================
Vergleicht gespeicherte Entscheidungen (data/decisions/log/) mit
den tatsächlichen Preisbewegungen in den nächsten 10 Tagen.

Frage: "Wenn das System am Tag X CRITICAL/WARNING für Ticker Y gesagt hat,
        ist der Preis in den nächsten 10 Tagen wirklich >5% gefallen?"

Verwendung:
  python src/backtesting/evaluate_log.py
  python src/backtesting/evaluate_log.py --min-days 10  (nur Logs die alt genug sind)

Output:
  Konsolenbericht mit Precision, Recall, F1 pro Severity
  data/backtesting/log_evaluation.parquet
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

LOG_DIR  = ROOT / "data/decisions/log"
DATA_DIR = ROOT / "data/detection"
OUT_DIR  = ROOT / "data/backtesting"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON      = 10    # Tage in die Zukunft (muss zu drawdown_probability.py passen)
DD_THRESHOLD = 0.05  # 5% Drawdown


def load_price_history() -> pd.DataFrame:
    """Lade alle OHLCV-Daten für Outcome-Berechnung."""
    dfs = []
    for f in sorted(DATA_DIR.glob("*.parquet")):
        if f.stem.startswith("^"):
            continue
        df = pd.read_parquet(f)[["Date", "Close"]].copy()
        df["ticker"] = f.stem
        df["Date"]   = pd.to_datetime(df["Date"])
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    return data.sort_values(["ticker", "Date"]).reset_index(drop=True)


def compute_actual_outcome(prices: pd.DataFrame, ticker: str, decision_date: pd.Timestamp) -> dict:
    """
    Berechne ob nach decision_date ein Drawdown > 5% in den nächsten HORIZON Tagen eingetreten ist.
    Gibt None zurück wenn nicht genug Daten vorhanden (zu früh).
    """
    hist = prices[prices["ticker"] == ticker].sort_values("Date")
    future = hist[hist["Date"] > decision_date].head(HORIZON)

    if len(future) < HORIZON:
        return None  # noch nicht genug Tage vergangen

    close_at_decision = hist[hist["Date"] <= decision_date]["Close"].iloc[-1]
    min_future        = future["Close"].min()
    end_future        = future["Close"].iloc[-1]

    dd       = (min_future - close_at_decision) / close_at_decision
    ret_10d  = (end_future - close_at_decision) / close_at_decision

    return {
        "drawdown_event": 1 if dd <= -DD_THRESHOLD else 0,
        "actual_drawdown": round(dd, 4),
        "actual_return_10d": round(ret_10d, 4),
    }


def run(min_days: int = HORIZON):
    log_files = sorted(LOG_DIR.glob("*.parquet"))
    if not log_files:
        print(f"Keine Log-Dateien in {LOG_DIR}")
        print("Starte die Decision Pipeline um Logs zu erzeugen:")
        print("  python src/decision/decision_pipeline.py")
        return

    today = pd.Timestamp.today().normalize()
    print(f"Lade Preishistorie...")
    prices = load_price_history()

    all_rows = []
    for log_file in log_files:
        log_date = pd.Timestamp(log_file.stem)
        days_ago = (today - log_date).days

        if days_ago < min_days:
            print(f"  {log_file.stem} — nur {days_ago} Tage alt, überspringe (brauche {min_days})")
            continue

        log_df = pd.read_parquet(log_file)
        print(f"  {log_file.stem} — {len(log_df)} Ticker, {days_ago} Tage alt")

        for _, row in log_df.iterrows():
            ticker   = row["ticker"]
            severity = row.get("severity", "NORMAL")
            p_dd     = float(row.get("p_drawdown", 0))

            outcome = compute_actual_outcome(prices, ticker, log_date)
            if outcome is None:
                continue

            all_rows.append({
                "log_date":         str(log_date.date()),
                "ticker":           ticker,
                "severity":         severity,
                "action":           row.get("action", ""),
                "p_drawdown":       round(p_dd, 4),
                "drawdown_risk":    row.get("drawdown_risk", ""),
                "anomaly_score":    row.get("anomaly_score", 0),
                **outcome,
            })

    if not all_rows:
        print(f"\nKeine auswertbaren Logs (brauche Logs die mindestens {min_days} Tage alt sind).")
        return

    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT_DIR / "log_evaluation.parquet", index=False)
    print(f"\nGespeichert: {OUT_DIR / 'log_evaluation.parquet'} ({len(df)} Einträge)")

    # ── Auswertung ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DECISION LOG EVALUATION")
    print(f"Horizont: {HORIZON} Tage  |  Schwelle: >{DD_THRESHOLD*100:.0f}% Drawdown")
    print("=" * 60)

    # Gesamt-Statistik
    total     = len(df)
    n_events  = df["drawdown_event"].sum()
    base_rate = n_events / total
    print(f"\nGesamt: {total} Entscheidungen  |  Echte Drawdowns: {n_events} ({base_rate:.1%} base rate)")

    # Pro Severity
    print("\nPräzision pro Severity:")
    print(f"{'Severity':<20} {'N':>5} {'Drawdowns':>9} {'Precision':>10} {'Avg DD':>8} {'Avg Ret 10d':>12}")
    print("-" * 65)
    for sev in ["CRITICAL", "WARNING", "WATCH", "NORMAL", "POSITIVE_SIGNAL", "ENTRY"]:
        sub = df[df["severity"] == sev]
        if len(sub) == 0:
            continue
        n_ev  = sub["drawdown_event"].sum()
        prec  = n_ev / len(sub)
        avg_dd  = sub["actual_drawdown"].mean()
        avg_ret = sub["actual_return_10d"].mean()
        print(f"{sev:<20} {len(sub):>5} {int(n_ev):>9} {prec:>9.1%} {avg_dd:>7.1%} {avg_ret:>11.1%}")

    # CRITICAL + WARNING zusammen (Alert-Precision)
    alerts = df[df["severity"].isin(["CRITICAL", "WARNING"])]
    no_alerts = df[~df["severity"].isin(["CRITICAL", "WARNING"])]
    if len(alerts) > 0:
        tp = alerts["drawdown_event"].sum()
        fp = len(alerts) - tp
        fn = no_alerts["drawdown_event"].sum() if len(no_alerts) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"\nCRITICAL + WARNING Alert-Qualität:")
        print(f"  Precision : {precision:.1%}  (von {len(alerts)} Alerts waren {int(tp)} richtig)")
        print(f"  Recall    : {recall:.1%}  (von {int(tp+fn)} echten Drawdowns wurden {int(tp)} erkannt)")
        print(f"  F1 Score  : {f1:.3f}")
        print(f"  False Positives: {int(fp)}  |  False Negatives: {int(fn)}")

    # Top False Positives (System hat CRITICAL gesagt aber nichts passiert)
    fp_rows = df[(df["severity"] == "CRITICAL") & (df["drawdown_event"] == 0)]
    if len(fp_rows) > 0:
        print(f"\nTop False Positives (CRITICAL, kein Drawdown eingetreten):")
        print(fp_rows[["log_date", "ticker", "p_drawdown", "actual_drawdown", "actual_return_10d"]]
              .sort_values("actual_return_10d", ascending=False)
              .head(10)
              .to_string(index=False))

    # Top True Positives
    tp_rows = df[(df["severity"].isin(["CRITICAL", "WARNING"])) & (df["drawdown_event"] == 1)]
    if len(tp_rows) > 0:
        print(f"\nTop True Positives (Alert war richtig):")
        print(tp_rows[["log_date", "ticker", "severity", "p_drawdown", "actual_drawdown"]]
              .sort_values("actual_drawdown")
              .head(10)
              .to_string(index=False))

    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-days", type=int, default=HORIZON,
                        help=f"Minimale Tage seit Log-Datum (default: {HORIZON})")
    args = parser.parse_args()
    run(min_days=args.min_days)
