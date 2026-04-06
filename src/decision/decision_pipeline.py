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
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")          # load FINNHUB_API_KEY and other secrets
sys.path.insert(0, str(ROOT / "src"))

from prediction.models.drawdown_probability  import load_data, predict as predict_drawdown
from prediction.models.meta_model           import predict_batch as meta_predict_batch, MODEL_PATH as META_MODEL_PATH
from prediction.features.technical_signals  import TECHNICAL_SIGNAL_COLS
from decision.decision_engine                import run_decision_engine
from explainability.finbert                  import fetch_news

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
    "excess_return", "volatility",
    # Trend & regime context (note: ma50/ma200 here are SPX MAs, not stock MAs)
    "Close", "regime", "volume_trend", "trend_strength",
]


def _fetch_live_sentiment(tickers: list) -> dict:
    """
    Fetch today's news and compute VADER + FinBERT + combined scores per ticker.
    Returns {ticker: {vader_score, finbert_score, news_sentiment_score}}.
    Skipped entirely if FINNHUB_API_KEY is missing.
    """
    import os, time
    if not os.environ.get("FINNHUB_API_KEY"):
        return {}

    from explainability.finbert import fetch_news as _fetch, aggregate_scores

    result = {}
    total  = len(tickers)
    for i, ticker in enumerate(tickers, 1):
        print(f"    sentiment [{i}/{total}] {ticker}...", end=" ", flush=True)
        headlines, _, timestamps = _fetch(ticker, limit=5, retries=1)
        time.sleep(1.0)
        if not headlines:
            print("no news")
            result[ticker] = {"vader_score": 0.0, "finbert_score": 0.0, "news_sentiment_score": 0.0}
            continue
        scores = aggregate_scores(headlines, timestamps, ticker=ticker)
        reasoning = f"  → {scores['groq_reasoning'][:55]}" if scores.get("groq_reasoning") else ""
        print(f"vader={scores['vader_score']:+.3f}  groq={scores['finbert_score']:+.3f}  combined={scores['news_sentiment_score']:+.3f}{reasoning}")
        result[ticker] = scores
    return result


def _load_valuation_signals() -> dict:
    """
    Load latest valuation metrics per ticker from data/fundamental/valuation.parquet.
    Returns dict: {ticker: {pe_ratio, pe_forward, pb_ratio, revenue_growth}}
    Falls back to empty dict if file doesn't exist yet.
    """
    path = ROOT / "data/fundamental/valuation.parquet"
    if not path.exists():
        print("  No valuation data found — run valuation_collector first.")
        return {}
    df = pd.read_parquet(path)
    df = df.sort_values("date").groupby("ticker").last().reset_index()
    result = {}
    for _, row in df.iterrows():
        result[row["ticker"]] = {
            "pe_ratio":       None if pd.isna(row.get("pe_ratio",       float("nan"))) else float(row["pe_ratio"]),
            "pe_forward":     None if pd.isna(row.get("pe_forward",     float("nan"))) else float(row["pe_forward"]),
            "pb_ratio":       None if pd.isna(row.get("pb_ratio",       float("nan"))) else float(row["pb_ratio"]),
            "revenue_growth": None if pd.isna(row.get("revenue_growth", float("nan"))) else float(row["revenue_growth"]),
        }
    print(f"  Valuation signals loaded for {len(result)} tickers.")
    return result


def _load_fundamental_signals() -> dict:
    """
    Load latest fundamental data per ticker from:
      data/fundamental/earnings.parquet  → days_to_next_earnings
      data/fundamental/insider.parquet   → insider_sentiment
      data/fundamental/options.parquet   → put_call_ratio, options_fear

    Returns dict: {ticker: {days_to_next_earnings, insider_sentiment, put_call_ratio, options_fear}}
    Falls back to empty dict if files don't exist yet (collectors not yet run).
    """
    fund_dir = ROOT / "data/fundamental"
    result: dict = {}

    def _latest(path: Path, cols: list) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        if "date" in df.columns:
            df = df.sort_values("date").groupby("ticker").last().reset_index()
        return df[["ticker"] + [c for c in cols if c in df.columns]]

    earnings = _latest(fund_dir / "earnings.parquet", ["days_to_next_earnings"])
    insider  = _latest(fund_dir / "insider.parquet",  ["insider_sentiment"])
    options  = _latest(fund_dir / "options.parquet",  ["put_call_ratio", "options_fear"])

    def _tickers(df): return df["ticker"].tolist() if "ticker" in df.columns else []
    all_tickers = set(_tickers(earnings) + _tickers(insider) + _tickers(options))

    for ticker in all_tickers:
        row: dict = {}
        if not earnings.empty and ticker in earnings["ticker"].values:
            v = earnings[earnings["ticker"] == ticker].iloc[0].get("days_to_next_earnings")
            row["days_to_next_earnings"] = None if pd.isna(v) else int(v)
        if not insider.empty and ticker in insider["ticker"].values:
            row["insider_sentiment"] = float(insider[insider["ticker"] == ticker].iloc[0].get("insider_sentiment", 0.0))
        if not options.empty and ticker in options["ticker"].values:
            opt_row = options[options["ticker"] == ticker].iloc[0]
            row["put_call_ratio"] = float(opt_row.get("put_call_ratio", 0.85))
            row["options_fear"]   = int(opt_row.get("options_fear", 0))
        result[ticker] = row

    if result:
        print(f"  Fundamental signals loaded for {len(result)} tickers.")
    else:
        print("  No fundamental data found — run earnings/insider/options collectors first.")
    return result


def _compute_stock_ma_positions(cutoff_date=None) -> dict:
    """
    Compute each stock's own 50-day and 200-day moving average position.
    Returns {ticker: {price_vs_ma200, price_vs_ma50}}.

    Note: ma50/ma200 columns in detection parquets are S&P 500 MAs (market context),
    NOT the individual stock's MAs. This function computes the stock-specific values.
    """
    result = {}
    for f in sorted(DATA_DIR.glob("*.parquet")):
        if f.stem.startswith("^"):
            continue
        df = pd.read_parquet(f)[["Date", "Close"]].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        if cutoff_date is not None:
            df = df[df["Date"] <= cutoff_date]
        if len(df) < 20:
            continue
        close = df["Close"]
        ma50  = close.rolling(50,  min_periods=20).mean().iloc[-1]
        ma200 = close.rolling(200, min_periods=50).mean().iloc[-1]
        latest_close = close.iloc[-1]
        result[f.stem] = {
            "price_vs_ma200": round((latest_close / ma200 - 1), 4) if ma200 > 0 else 0.0,
            "price_vs_ma50":  round((latest_close / ma50  - 1), 4) if ma50  > 0 else 0.0,
        }
    return result


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

    # ── 1. Load detection data
    print("\n[1/4] Loading detection data...")
    data = load_data()

    # ── 2. Drawdown Probability Model (replaces Risk + Direction)
    print("[2/4] Running Drawdown Probability Model...")
    drawdown_df = predict_drawdown(data)   # ticker, p_drawdown, drawdown_risk

    # ── 3. Load latest detection signals
    print("[3/4] Loading latest detection signals...")
    detection_df = _load_latest_detection(cutoff)

    # ── 3e. Compute stock-specific MA positions (not SPX MAs)
    print("[3e/4] Computing stock MA positions...")
    ma_positions = _compute_stock_ma_positions(cutoff)

    # ── 3f. Extract technical signals from loaded data (computed by add_technical_signals)
    print("[3f/4] Extracting technical signals from loaded data...")
    avail_tech  = [c for c in TECHNICAL_SIGNAL_COLS if c in data.columns]
    tech_latest = data.groupby("ticker").last().reset_index()
    tech_signals = {
        row["ticker"]: {col: row[col] for col in avail_tech}
        for _, row in tech_latest.iterrows()
    }
    print(f"  Technical signals extracted for {len(tech_signals)} tickers.")

    # ── 4. Merge
    merged = drawdown_df.merge(detection_df, on="ticker")

    # Extract latest VIX per ticker from the enriched data (has vix_level from FRED)
    vix_map = (
        data.groupby("ticker")["vix_level"].last()
        if "vix_level" in data.columns
        else {}
    )
    vix_change_map = (
        data.groupby("ticker")["vix_change"].last()
        if "vix_change" in data.columns
        else {}
    )

    # ── 4b. Load fundamental signals (earnings, insider, options)
    print("[3b/4] Loading fundamental signals...")
    fundamental = _load_fundamental_signals()

    # ── 4c. Load valuation signals (P/E, P/B, revenue growth)
    print("[3c/4] Loading valuation signals...")
    valuation = _load_valuation_signals()

    # ── 4d. Fetch live news sentiment
    print("[3d/4] Fetching live news sentiment...")
    sentiment_scores = _fetch_live_sentiment(merged["ticker"].tolist())
    if not sentiment_scores:
        print("  (skipped — no FINNHUB_API_KEY)")

    # ── 5. Build records for Decision Engine
    records = []
    for _, row in merged.iterrows():
        records.append({
            "ticker":               row["ticker"],
            "date":                 str(row["Date"].date()),
            "p_drawdown":           float(row["p_drawdown"]),
            "drawdown_risk":        row["drawdown_risk"],
            "anomaly_score":        int(row["anomaly_score"]),
            "anomaly_score_weighted": float(row.get("anomaly_score_weighted", row["anomaly_score"] / 4.0)),
            "market_anomaly":       bool(row["market_anomaly"]),
            "sector_anomaly":       bool(row["sector_anomaly"]),
            "es_ratio":             float(row["es_ratio"])         if pd.notna(row["es_ratio"])         else 0.0,
            "rsi":                  float(row["rsi"])               if pd.notna(row["rsi"])               else 50.0,
            "momentum_5":           float(row["momentum_5"])       if pd.notna(row["momentum_5"])       else 0.0,
            "momentum_10":          float(row["momentum_10"])      if pd.notna(row["momentum_10"])      else 0.0,
            "drawdown":             float(row["max_drawdown_30d"]) if pd.notna(row["max_drawdown_30d"]) else 0.0,
            "excess_return":        float(row["excess_return"])    if pd.notna(row["excess_return"])    else 0.0,
            "obv_signal":           float(row["obv_signal"])       if pd.notna(row["obv_signal"])       else 0.0,
            "volatility":           float(row["volatility"])       if pd.notna(row["volatility"])       else 0.02,
            "vader_score":          sentiment_scores.get(row["ticker"], {}).get("vader_score",          0.0),
            "finbert_score":        sentiment_scores.get(row["ticker"], {}).get("finbert_score",        0.0),
            "news_sentiment_score": sentiment_scores.get(row["ticker"], {}).get("news_sentiment_score", 0.0),
            "vix_level":            float(vix_map.get(row["ticker"], 20.0)),
            # Fundamental signals
            "days_to_next_earnings": fundamental.get(row["ticker"], {}).get("days_to_next_earnings", None),
            "insider_sentiment":     fundamental.get(row["ticker"], {}).get("insider_sentiment", 0.0),
            "put_call_ratio":        fundamental.get(row["ticker"], {}).get("put_call_ratio", 0.85),
            "options_fear":          fundamental.get(row["ticker"], {}).get("options_fear", 0),
            # Valuation signals
            "pe_ratio":       valuation.get(row["ticker"], {}).get("pe_ratio",       None),
            "pe_forward":     valuation.get(row["ticker"], {}).get("pe_forward",     None),
            "pb_ratio":       valuation.get(row["ticker"], {}).get("pb_ratio",       None),
            "revenue_growth": valuation.get(row["ticker"], {}).get("revenue_growth", None),
            # Trend & regime context (stock-specific MAs from _compute_stock_ma_positions)
            "price_vs_ma200": ma_positions.get(row["ticker"], {}).get("price_vs_ma200", 0.0),
            "price_vs_ma50":  ma_positions.get(row["ticker"], {}).get("price_vs_ma50",  0.0),
            "regime":         str(row.get("regime", "unknown")) if pd.notna(row.get("regime")) else "unknown",
            "volume_trend":   float(row["volume_trend"]) if pd.notna(row.get("volume_trend")) else 1.0,
            "trend_strength": float(row["trend_strength"]) if pd.notna(row.get("trend_strength")) else 0.0,
            "vix_change":          float(vix_change_map.get(row["ticker"], 0.0)),
            # Bullish technical signals
            "price_vs_ema20":           tech_signals.get(row["ticker"], {}).get("price_vs_ema20",           0.0),
            "golden_cross":             tech_signals.get(row["ticker"], {}).get("golden_cross",             False),
            "rsi_oversold_bounce":      tech_signals.get(row["ticker"], {}).get("rsi_oversold_bounce",      False),
            "macd_cross_bullish":       tech_signals.get(row["ticker"], {}).get("macd_cross_bullish",       False),
            "hh_hl":                    tech_signals.get(row["ticker"], {}).get("hh_hl",                    False),
            "volume_breakout":          tech_signals.get(row["ticker"], {}).get("volume_breakout",          False),
            # Bearish technical signals
            "death_cross":              tech_signals.get(row["ticker"], {}).get("death_cross",              False),
            "ll_lh":                    tech_signals.get(row["ticker"], {}).get("ll_lh",                    False),
            "macd_cross_bearish":       tech_signals.get(row["ticker"], {}).get("macd_cross_bearish",       False),
            "panic_volume":             tech_signals.get(row["ticker"], {}).get("panic_volume",             False),
            "volume_spike_no_recovery": tech_signals.get(row["ticker"], {}).get("volume_spike_no_recovery", False),
            "rsi_below_50_high_vol":    bool(tech_signals.get(row["ticker"], {}).get("rsi_below_45_high_vol", False)),
            "rsi_oversold_no_bounce":   tech_signals.get(row["ticker"], {}).get("rsi_oversold_no_bounce",   False),
            # Neutral / WATCH signals
            "low_volume":                  tech_signals.get(row["ticker"], {}).get("low_volume",                  False),
            "consolidation":               tech_signals.get(row["ticker"], {}).get("consolidation",               False),
            "consolidation_above_ma50":    tech_signals.get(row["ticker"], {}).get("consolidation_above_ma50",    False),
            # Divergence
            "rsi_divergence_bullish":      tech_signals.get(row["ticker"], {}).get("rsi_divergence_bullish",      False),
        })

    # ── 6. Run Decision Engine
    print(f"[4/4] Running Decision Engine on {len(records)} tickers...")
    decisions = run_decision_engine(records)

    # ── 7. Save to parquet
    out_df = pd.DataFrame([vars(d) for d in decisions])
    # Carry AnomalyInput context fields (valuation, MA, regime) for dashboard display
    context_rows = [
        {
            "ticker":               r["ticker"],
            "pe_ratio":             r.get("pe_ratio"),
            "pe_forward":           r.get("pe_forward"),
            "pb_ratio":             r.get("pb_ratio"),
            "revenue_growth":       r.get("revenue_growth"),
            "price_vs_ma200":       r.get("price_vs_ma200", 0.0),
            "price_vs_ma50":        r.get("price_vs_ma50",  0.0),
            "regime":               r.get("regime", "unknown"),
            "volume_trend":         r.get("volume_trend", 1.0),
            "trend_strength":       r.get("trend_strength", 0.0),
            "price_vs_ema20":            r.get("price_vs_ema20",           0.0),
            "golden_cross":              r.get("golden_cross",             False),
            "rsi_oversold_bounce":       r.get("rsi_oversold_bounce",      False),
            "macd_cross_bullish":        r.get("macd_cross_bullish",       False),
            "hh_hl":                     r.get("hh_hl",                    False),
            "volume_breakout":           r.get("volume_breakout",          False),
            "death_cross":               r.get("death_cross",              False),
            "ll_lh":                     r.get("ll_lh",                    False),
            "macd_cross_bearish":        r.get("macd_cross_bearish",       False),
            "panic_volume":              r.get("panic_volume",             False),
            "volume_spike_no_recovery":  r.get("volume_spike_no_recovery", False),
            "rsi_below_50_high_vol":     r.get("rsi_below_50_high_vol",    False),
            "rsi_oversold_no_bounce":    r.get("rsi_oversold_no_bounce",   False),
            "low_volume":                  r.get("low_volume",                  False),
            "consolidation":               r.get("consolidation",               False),
            "consolidation_above_ma50":    r.get("consolidation_above_ma50",    False),
            "rsi_divergence_bullish":      r.get("rsi_divergence_bullish",      False),
        }
        for r in records
    ]
    ctx_df = pd.DataFrame(context_rows)
    out_df = out_df.merge(ctx_df, on="ticker", how="left")

    # DecisionOutput already has p_drawdown, anomaly_score, drawdown_risk — no re-merge needed
    # Carry extra detection signals for portfolio page (only cols not already in out_df)
    det_cols  = ["rsi", "momentum_5", "momentum_10", "obv_signal",
                 "max_drawdown_30d", "market_anomaly", "sector_anomaly",
                 "es_ratio", "volatility", "excess_return"]
    det_avail = [c for c in det_cols if c in detection_df.columns]
    existing  = set(out_df.columns)
    det_new   = [c for c in det_avail if c not in existing]
    if det_new:
        out_df = out_df.merge(detection_df[["ticker"] + det_new], on="ticker", how="left")
    if "max_drawdown_30d" in out_df.columns:
        out_df.rename(columns={"max_drawdown_30d": "drawdown"}, inplace=True)

    # Carry sentiment scores through for the narrator
    sentiment_df = pd.DataFrame([
        {"ticker": t, **s} for t, s in sentiment_scores.items()
    ]) if sentiment_scores else pd.DataFrame(columns=["ticker", "vader_score", "finbert_score", "news_sentiment_score"])
    if not sentiment_df.empty:
        out_df = out_df.merge(sentiment_df, on="ticker", how="left")
        for col in ["vader_score", "finbert_score", "news_sentiment_score"]:
            out_df[col] = out_df[col].fillna(0.0)
        # Stamp when news was fetched so dashboard can warn if stale
        out_df["sentiment_fetched_at"] = pd.Timestamp.now().isoformat()
    else:
        out_df["sentiment_fetched_at"] = None

    # ── 7b. Meta-Model — refined drawdown probability (stacking layer)
    if META_MODEL_PATH.exists():
        print("  Running Meta-Model (stacking)...")
        out_df = meta_predict_batch(out_df, detection_df=detection_df)
        # Update confidence: use meta probability directly as confidence for risk severities
        for col in ["p_drawdown_meta", "meta_confidence"]:
            if col in out_df.columns:
                out_df[col] = out_df[col].round(4)
        print(f"  Meta-Model done. Avg p_drawdown_meta: {out_df['p_drawdown_meta'].mean():.1%}")
    else:
        out_df["p_drawdown_meta"] = out_df["p_drawdown"]
        out_df["meta_confidence"] = 0.0

    out_path = OUT_DIR / "decisions.parquet"
    out_df.to_parquet(out_path, index=False)

    # ── 7c. Decision Log — save dated snapshot for backtesting (never overwritten)
    log_dir  = OUT_DIR / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_date = cutoff.date() if cutoff else pd.Timestamp.today().date()
    log_path = log_dir / f"{run_date}.parquet"
    if log_path.exists():
        print(f"  Log for {run_date} already exists — skipping (delete manually to overwrite).")
    else:
        out_df["run_date"] = str(run_date)
        out_df.to_parquet(log_path, index=False)
        print(f"  Decision log saved: {log_path}")

    # ── 8. Print summary
    print(f"\nSaved: {out_path}")
    meta_map = dict(zip(out_df["ticker"], out_df.get("p_drawdown_meta", out_df["p_drawdown"])))
    print(f"\n{'Ticker':<8} {'Severity':<20} {'Action':<10} {'Conf':>5} {'Meta':>5}  Context")
    print("-" * 72)
    for d in sorted(decisions, key=lambda x: x.severity):
        meta = meta_map.get(d.ticker, d.p_drawdown)
        print(f"{d.ticker:<8} {d.severity:<20} {d.action:<10} {d.confidence:>4.0%} {meta:>4.0%}  {d.context}")

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
