"""
FinWatch AI — Layer 7: Explainability Builder (Orchestrator)
=============================================================
Runs 7A → 7B → 7C in sequence and saves the result.

  7A: xai.py             — SHAP values per ticker
  7B: narrative_engine.py — Signal alignment / conflict detection
  7C: llm_narrator.py    — Plain-English summary (placeholder)

Output: data/explanations/explanations.parquet
"""

import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from prediction.models.xgboost_risk import load_data, FEATURES
import explainability.xai as xai
import explainability.narrative_engine as narrative_engine

DECISION_IN = ROOT / "data/decisions/decisions.parquet"
DETECTION_DIR = ROOT / "data/detection"
OUT_DIR     = ROOT / "data/explanations"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_obv() -> dict[str, float]:
    """Latest OBV signal per ticker from detection parquets."""
    obv_map = {}
    for f in sorted(DETECTION_DIR.glob("*.parquet")):
        if f.stem.startswith("^"):
            continue
        df = pd.read_parquet(f)
        if "obv_signal" not in df.columns:
            obv_map[f.stem] = 0.0
            continue
        df["Date"] = pd.to_datetime(df["Date"])
        obv_map[f.stem] = float(df.sort_values("Date").iloc[-1]["obv_signal"])
    return obv_map


def run() -> pd.DataFrame:
    print("=" * 55)
    print("EXPLAINABILITY BUILDER — Layer 7")
    print("=" * 55)

    # ── 7A: SHAP ──────────────────────────────────────────────
    print("\n[1/4] Loading data + computing SHAP values...")
    data = load_data()
    latest, shap_matrix = xai.compute(data, FEATURES)
    tickers = latest["ticker"].tolist()
    print(f"      SHAP shape: {shap_matrix.shape}")

    # ── OBV + Decisions ───────────────────────────────────────
    print("[2/4] Loading OBV signals...")
    obv_map = _load_obv()

    print("[3/4] Loading Decision Engine output...")
    if not DECISION_IN.exists():
        raise FileNotFoundError(
            f"decisions.parquet not found at {DECISION_IN}\n"
            "Run decision_pipeline.py first."
        )
    decisions = pd.read_parquet(DECISION_IN)[
        ["ticker", "date", "severity", "action", "confidence", "context"]
    ]

    # ── 7B: Narrative Engine ──────────────────────────────────
    print("[4/4] Building narratives...")
    rows = []
    for i, ticker in enumerate(tickers):
        dec_row = decisions[decisions["ticker"] == ticker]
        if dec_row.empty:
            print(f"  Skipping {ticker} — no decision found")
            continue

        dec      = dec_row.iloc[0]
        shap_row = shap_matrix[i]
        top3     = xai.top3(shap_row, FEATURES)

        driver_name, driver_shap = top3[0]
        obv      = obv_map.get(ticker, 0.0)
        confirm  = narrative_engine.obv_label(obv)
        conflict = narrative_engine.detect_conflict(dec["severity"], obv)
        narrative_key, narrative_text = narrative_engine.build(
            dec["severity"], obv, driver_shap
        )

        row = {
            "ticker":           ticker,
            "date":             dec["date"],
            "severity":         dec["severity"],
            "action":           dec["action"],
            "confidence":       round(float(dec["confidence"]), 4),
            "driver":           driver_name,
            "driver_shap":      round(float(driver_shap), 4),
            "top3_shap":        xai.format_top3(top3),
            "obv_signal":       round(obv, 4),
            "confirmation":     confirm,
            "conflict":         conflict,
            "narrative":        narrative_key,
            "narrative_text":   narrative_text,
            "decision_context": dec["context"],
        }

        rows.append(row)

    result = pd.DataFrame(rows).sort_values("severity").reset_index(drop=True)

    out_path = OUT_DIR / "explanations.parquet"
    result.to_parquet(out_path, index=False)

    # ── Print summary ─────────────────────────────────────────
    print(f"\nSaved: {out_path}")
    print(f"\n{'Ticker':<8} {'Severity':<20} {'Driver':<18} {'OBV':>6}  Narrative")
    print("-" * 80)
    for _, r in result.iterrows():
        conflict_flag = " [!CONFLICT]" if r["conflict"] else ""
        print(
            f"{r['ticker']:<8} {r['severity']:<20} {r['driver']:<18} "
            f"{r['obv_signal']:>6.2f}  {r['narrative']}{conflict_flag}"
        )

    print("\n" + "=" * 55)
    print("Explainability Builder complete.")
    print("=" * 55)
    return result


if __name__ == "__main__":
    run()
