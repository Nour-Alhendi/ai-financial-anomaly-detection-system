"""
FinWatch AI — Backtesting Engine
==================================
Walk-forward backtest for the Decision Engine.

What it measures:
  - Precision: when system says CRITICAL/WARNING, how often was there a real drawdown?
  - Recall: of all real drawdowns, how many did the system catch?
  - Return after signal: average return in the 20 days following each signal
  - False alarm rate: how often CRITICAL fires but nothing happens

Walk-forward setup:
  Train window : 4 years rolling
  Test window  : 6 months
  Step         : 6 months
  → No lookahead bias

Anomaly Precision Backtesting (Option A):
  Isolation Forest and LSTM-AE are unsupervised — they have no labels.
  This module evaluates them post-hoc using anomaly_score_weighted (which
  gives IF and LSTM-AE 0.30 weight each vs 0.20 for z-scores) and raw
  ae_error percentile buckets. We measure precision/recall against real
  drawdown events. This answers: "When IF + LSTM-AE both fired (score=0.60),
  how often was there actually a >5% drawdown?"
  → Saved to: data/backtesting/anomaly_precision.parquet

  Option B (not implemented): The LSTM-AE reconstruction error (ae_error)
  could be used as a supervised feature by training a thin classifier on top
  of it, using drawdown labels as targets. Since it's an LSTM-AE (sequence
  model), it already captures temporal patterns — making it well-suited for
  this. The ae_error would replace the raw anomaly threshold with a learned
  one. Skipped here because the LSTM-AE is already feeding ae_error into
  the Meta-Model, which learns the optimal threshold indirectly.

  Option C (already covered): VIX-context weighting of anomaly signals
  (anomaly at VIX=12 vs VIX=35 should have different weight) is handled
  by the Meta-Model, which receives both anomaly_score_weighted and
  vix_level/vix_high as features and learns the interaction automatically.
  No separate implementation needed.

Output:
  data/backtesting/backtest_results.parquet   — per-signal results
  data/backtesting/signal_precision.parquet   — precision per signal type (→ used as Confidence)
  data/backtesting/anomaly_precision.parquet  — precision per anomaly detector level/bucket
  data/backtesting/summary.txt                — human-readable report
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = ROOT / "data/detection"
OUT_DIR  = ROOT / "data/backtesting"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON      = 20     # days forward for outcome measurement
DD_THRESHOLD = 0.05   # 5% drawdown = "risk event happened"
TRAIN_YEARS  = 4
STEP_MONTHS  = 6

# Signal definitions — what constitutes each signal type
# Based on DecisionOutput fields
SIGNAL_THRESHOLDS = {
    "strong_anomaly":    lambda r: r.get("anomaly_score_weighted", 0) >= 0.50,
    "medium_anomaly":    lambda r: 0.30 <= r.get("anomaly_score_weighted", 0) < 0.50,
    "high_p_drawdown":   lambda r: r.get("p_drawdown", 0) >= 0.60,
    "medium_p_drawdown": lambda r: 0.45 <= r.get("p_drawdown", 0) < 0.60,
    "rsi_overbought":    lambda r: r.get("rsi", 50) >= 70,
    "rsi_oversold":      lambda r: r.get("rsi", 50) <= 30,
    "negative_momentum": lambda r: r.get("momentum_5", 0) < -0.02 and r.get("momentum_10", 0) < -0.01,
    "negative_news":     lambda r: r.get("news_sentiment_score", 0) <= -0.10,
    "severity_critical": lambda r: r.get("severity", "") == "CRITICAL",
    "severity_warning":  lambda r: r.get("severity", "") in ("CRITICAL", "WARNING"),
}


def load_all_detection() -> pd.DataFrame:
    dfs = []
    for f in sorted(DATA_DIR.glob("*.parquet")):
        if f.stem.startswith("^"):
            continue
        df = pd.read_parquet(f)
        df["ticker"] = f.stem
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    data["Date"] = pd.to_datetime(data["Date"])
    return data.sort_values(["ticker", "Date"]).reset_index(drop=True)


def compute_outcomes(data: pd.DataFrame) -> pd.DataFrame:
    """Add future_drawdown_event column: 1 if max drawdown > DD_THRESHOLD in next HORIZON days."""
    def _outcomes(df):
        df = df.copy().sort_values("Date")
        closes = df["Close"].values
        events = []
        returns_20d = []
        for i in range(len(closes)):
            if i + HORIZON < len(closes):
                window = closes[i + 1: i + HORIZON + 1]
                dd = (window.min() - closes[i]) / closes[i]
                ret = (closes[i + HORIZON] - closes[i]) / closes[i]
                events.append(1 if dd <= -DD_THRESHOLD else 0)
                returns_20d.append(round(ret, 4))
            else:
                events.append(np.nan)
                returns_20d.append(np.nan)
        df["future_drawdown_event"] = events
        df["future_return_20d"]     = returns_20d
        return df

    return data.groupby("ticker", group_keys=False).apply(_outcomes)


def add_vix(data: pd.DataFrame) -> pd.DataFrame:
    try:
        from pandas_datareader import data as web
        vix = web.DataReader("VIXCLS", "fred", "2015-01-01", "2030-12-31").reset_index()
        vix.columns = ["Date", "vix_level"]
        vix["Date"] = pd.to_datetime(vix["Date"])
    except Exception:
        vix = pd.DataFrame({"Date": pd.date_range("2015-01-01", "2030-12-31"), "vix_level": 20.0})
    vix["vix_change"] = vix["vix_level"].pct_change().fillna(0)
    vix["vix_high"]   = (vix["vix_level"] > 25).astype(int)
    data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)
    return data.merge(vix[["Date", "vix_level", "vix_change", "vix_high"]], on="Date", how="left").assign(
        vix_level  = lambda d: d["vix_level"].ffill().fillna(20),
        vix_change = lambda d: d["vix_change"].ffill().fillna(0),
        vix_high   = lambda d: d["vix_high"].ffill().fillna(0),
    )


def _ma_position_at(ticker: str, as_of_date, data: pd.DataFrame) -> float:
    """Return (close / MA200 - 1) for a ticker using price history up to as_of_date."""
    hist = data[(data["ticker"] == ticker) & (data["Date"] < as_of_date)].sort_values("Date")
    if len(hist) < 50:
        return 0.0
    close  = hist["Close"]
    ma200  = close.rolling(200, min_periods=50).mean().iloc[-1]
    latest = close.iloc[-1]
    return round((latest / ma200 - 1), 4) if ma200 > 0 else 0.0


def run_walk_forward(data: pd.DataFrame) -> pd.DataFrame:
    """
    Walk-forward backtest.
    For each test window, simulate what the decision engine would have said
    using only data available at that time, then compare to actual outcomes.
    """
    from prediction.models.drawdown_probability import _prep_X, FEATURES, MODEL_PATH
    from decision.decision_engine import run_decision_engine, AnomalyInput
    import joblib

    dates = pd.date_range(
        start=data["Date"].min() + pd.DateOffset(years=TRAIN_YEARS),
        end=data["Date"].max()   - pd.DateOffset(days=HORIZON),
        freq=f"{STEP_MONTHS}MS",
    )

    all_results = []

    for test_start in dates:
        test_end   = test_start + pd.DateOffset(months=STEP_MONTHS)
        train_end  = test_start
        train_start = test_start - pd.DateOffset(years=TRAIN_YEARS)

        train_data = data[(data["Date"] >= train_start) & (data["Date"] < train_end)].copy()
        test_data  = data[(data["Date"] >= test_start)  & (data["Date"] < test_end)].copy()

        if train_data.empty or test_data.empty:
            continue

        # ── Train drawdown model on this window ──────────────────────────────
        from xgboost import XGBClassifier

        def _label(df):
            df = df.copy().sort_values("Date")
            closes = df["Close"].values
            labels = []
            for i in range(len(closes)):
                if i + HORIZON < len(closes):
                    w  = closes[i + 1: i + HORIZON + 1]
                    dd = (w.min() - closes[i]) / closes[i]
                    labels.append(1 if dd <= -DD_THRESHOLD else 0)
                else:
                    labels.append(np.nan)
            df["drawdown_event"] = labels
            return df

        train_labeled = train_data.groupby("ticker", group_keys=False).apply(_label)
        train_labeled = train_labeled.dropna(subset=["drawdown_event"])

        avail_features = [f for f in FEATURES if f in train_labeled.columns]
        train_labeled = train_labeled.copy()
        X_tr = train_labeled[avail_features].apply(lambda col: pd.to_numeric(col, errors="coerce")).fillna(0)
        y_tr = train_labeled["drawdown_event"].astype(int)

        if y_tr.sum() < 10:
            continue

        pos_rate = y_tr.mean()
        spw = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

        wf_model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            scale_pos_weight=spw,
            objective="binary:logistic",
            eval_metric="auc", random_state=42, n_jobs=-1,
        )
        wf_model.fit(X_tr, y_tr, verbose=0)

        # ── Predict on test window ───────────────────────────────────────────
        # Latest row per ticker in each test month
        test_latest = (
            test_data.sort_values("Date")
            .groupby("ticker")
            .last()
            .reset_index()
        )

        X_te = test_latest[avail_features].apply(lambda col: pd.to_numeric(col, errors="coerce")).fillna(0)
        p_drawdown = wf_model.predict_proba(X_te)[:, 1]

        # ── Build decision records ───────────────────────────────────────────
        records = []
        for i, row in test_latest.iterrows():
            p_dd = float(p_drawdown[test_latest.index.get_loc(i)])
            aw   = float(row.get("anomaly_score_weighted",
                                  row.get("anomaly_score", 0) / 4.0))
            records.append({
                "ticker":                 row["ticker"],
                "date":                   str(row["Date"].date()),
                "p_drawdown":             p_dd,
                "drawdown_risk":          "high" if p_dd >= 0.45 else "low",
                "anomaly_score":          int(row.get("anomaly_score", 0)),
                "anomaly_score_weighted": aw,
                "market_anomaly":         bool(row.get("market_anomaly", False)),
                "sector_anomaly":         bool(row.get("sector_anomaly", False)),
                "rsi":                    float(row.get("rsi", 50)),
                "momentum_5":             float(row.get("momentum_5", 0)),
                "momentum_10":            float(row.get("momentum_10", 0)),
                "drawdown":               float(row.get("max_drawdown_30d", 0)),
                "excess_return":          float(row.get("excess_return", 0)),
                "obv_signal":             float(row.get("obv_signal", 0)),
                "volatility":             float(row.get("volatility", 0.02)),
                "es_ratio":               float(row.get("es_ratio", 1.0)),
                "vader_score":            0.0,
                "finbert_score":          0.0,
                "news_sentiment_score":   0.0,
                "vix_level":              float(row.get("vix_level", 20.0)),
                # Valuation — not available historically, defaults to None (no gate applied)
                "pe_ratio":       None,
                "pe_forward":     None,
                "pb_ratio":       None,
                "revenue_growth": None,
                # Trend & regime context
                # price_vs_ma200: computed from stock's own price history up to test_start
                "price_vs_ma200": _ma_position_at(row["ticker"], test_start, data),
                "price_vs_ma50":  0.0,  # not used in gates
                "regime":         str(row.get("regime", "unknown")) if pd.notna(row.get("regime", None)) else "unknown",
                "volume_trend":   float(row["volume_trend"]) if pd.notna(row.get("volume_trend")) else 1.0,
                "trend_strength": float(row["trend_strength"]) if pd.notna(row.get("trend_strength")) else 0.0,
            })

        decisions = run_decision_engine(records)

        # ── Match decisions to outcomes ──────────────────────────────────────
        # Use the future outcome for the SPECIFIC date of the decision (last row
        # per ticker in the test window), not the worst-case over the whole window.
        outcome_map = {}
        for i, row in test_latest.iterrows():
            ticker   = row["ticker"]
            dec_date = row["Date"]
            # Find this exact row in data (which has future outcomes computed)
            match = data[
                (data["ticker"] == ticker) &
                (data["Date"] == dec_date)
            ]
            if not match.empty and not pd.isna(match.iloc[0]["future_drawdown_event"]):
                outcome_map[ticker] = {
                    "event":     int(match.iloc[0]["future_drawdown_event"]),
                    "return_20": float(match.iloc[0]["future_return_20d"]),
                }

        for dec in decisions:
            if dec.ticker not in outcome_map:
                continue
            outcome = outcome_map[dec.ticker]
            row_dict = vars(dec)
            row_dict["test_start"]           = test_start
            row_dict["actual_drawdown_event"] = outcome["event"]
            row_dict["actual_return_20d"]     = outcome["return_20"]
            row_dict["true_positive"]         = int(
                dec.severity in ("CRITICAL", "WARNING") and outcome["event"] == 1
            )
            row_dict["false_positive"]        = int(
                dec.severity in ("CRITICAL", "WARNING") and outcome["event"] == 0
            )
            row_dict["true_negative"]         = int(
                dec.severity not in ("CRITICAL", "WARNING") and outcome["event"] == 0
            )
            row_dict["false_negative"]        = int(
                dec.severity not in ("CRITICAL", "WARNING") and outcome["event"] == 1
            )
            all_results.append(row_dict)

        period = f"{test_start.date()} → {test_end.date()}"
        n_crit = sum(1 for d in decisions if d.severity == "CRITICAL")
        n_warn = sum(1 for d in decisions if d.severity == "WARNING")
        print(f"  {period}  |  CRITICAL={n_crit}  WARNING={n_warn}  tickers={len(decisions)}")

    return pd.DataFrame(all_results)


def compute_signal_precision(results: pd.DataFrame) -> pd.DataFrame:
    """
    For each signal type: precision, recall, avg_return_after_signal.
    This becomes the foundation for data-driven Confidence.
    """
    rows = []
    for signal_name, signal_fn in SIGNAL_THRESHOLDS.items():
        # Apply signal function to results
        fired = results.apply(
            lambda r: signal_fn({
                "anomaly_score_weighted": r.get("anomaly_score_weighted", 0),
                "p_drawdown":             r.get("p_drawdown", 0),
                "rsi":                    r.get("rsi", 50),
                "momentum_5":             r.get("momentum_5", 0),
                "momentum_10":            r.get("momentum_10", 0),
                "news_sentiment_score":   r.get("news_sentiment_score", 0),
                "severity":               r.get("severity", "NORMAL"),
            }), axis=1
        )
        signal_rows    = results[fired]
        no_signal_rows = results[~fired]

        if len(signal_rows) == 0:
            continue

        tp = signal_rows["actual_drawdown_event"].sum()
        fp = len(signal_rows) - tp
        fn = no_signal_rows["actual_drawdown_event"].sum() if len(no_signal_rows) > 0 else 0

        precision      = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall         = tp / (tp + fn) if (tp + fn) > 0 else 0
        avg_ret        = signal_rows["actual_return_20d"].mean()
        avg_ret_no_sig = no_signal_rows["actual_return_20d"].mean() if len(no_signal_rows) > 0 else 0

        rows.append({
            "signal":           signal_name,
            "n_fired":          len(signal_rows),
            "n_true_positive":  int(tp),
            "n_false_positive": int(fp),
            "precision":        round(precision, 3),
            "recall":           round(recall, 3),
            "f1":               round(2 * precision * recall / (precision + recall), 3)
                                if (precision + recall) > 0 else 0,
            "avg_return_after_signal":    round(avg_ret, 4),
            "avg_return_no_signal":       round(avg_ret_no_sig, 4),
            "return_edge":                round(avg_ret - avg_ret_no_sig, 4),
        })

    return pd.DataFrame(rows).sort_values("precision", ascending=False)


def compute_anomaly_precision(data: pd.DataFrame) -> pd.DataFrame:
    """
    Option A — Labeled Anomaly Backtesting with Validation / Holdout Split.

    Evaluates anomaly_score_weighted and ae_error against real drawdown labels.

    Split:
      Validation : 2022-01-01 → 2024-01-01  ← threshold selection happens here
      Holdout    : 2024-01-01 → 2026-03-01  ← clean evaluation, never used for tuning

    This prevents threshold overfitting: the threshold picked on validation
    is evaluated on truly unseen holdout data.

    Returns a DataFrame saved to data/backtesting/anomaly_precision.parquet.
    Column 'split' indicates "validation" or "holdout".
    """
    VALIDATION_START = pd.Timestamp("2022-01-01")
    HOLDOUT_START    = pd.Timestamp("2024-01-01")
    HOLDOUT_END      = pd.Timestamp("2026-03-01")

    required = {"future_drawdown_event", "future_return_20d", "anomaly_score"}
    if not required.issubset(data.columns):
        print("  Skipping anomaly precision: missing required columns")
        return pd.DataFrame()

    labeled      = data.dropna(subset=["future_drawdown_event", "future_return_20d"]).copy()
    val_data     = labeled[(labeled["Date"] >= VALIDATION_START) & (labeled["Date"] < HOLDOUT_START)]
    holdout_data = labeled[(labeled["Date"] >= HOLDOUT_START)    & (labeled["Date"] < HOLDOUT_END)]

    rows = []

    def _eval_threshold(subset, fired_mask, baseline):
        signal_rows    = subset[fired_mask]
        no_signal_rows = subset[~fired_mask]
        if len(signal_rows) == 0:
            return None
        tp = signal_rows["future_drawdown_event"].astype(int).sum()
        fp = len(signal_rows) - tp
        fn = no_signal_rows["future_drawdown_event"].astype(int).sum() if len(no_signal_rows) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        return {
            "n_fired":   int(fired_mask.sum()),
            "n_total":   len(subset),
            "fire_rate": round(fired_mask.mean(), 3),
            "precision": round(prec, 3),
            "recall":    round(rec, 3),
            "f1":        round(f1, 3),
            "avg_return_after_signal": round(signal_rows["future_return_20d"].mean(), 4),
            "avg_return_baseline":     round(baseline, 4),
            "return_edge": round(signal_rows["future_return_20d"].mean() - baseline, 4),
        }

    # ── anomaly_score_weighted ───────────────────────────────────────────────
    # Weights: z_short=0.20, z_long=0.20, IF=0.30, LSTM-AE=0.30
    # Threshold reference:
    #   0.30 = IF alone, or LSTM-AE alone
    #   0.50 = IF + z_short, or LSTM-AE + z_short
    #   0.60 = IF + LSTM-AE, or IF + both z-scores
    #   0.80 = IF + LSTM-AE + one z-score
    #   1.00 = all 4 fired
    if "anomaly_score_weighted" in labeled.columns:
        for threshold in [0.30, 0.50, 0.60, 0.80, 1.00]:
            for split_name, subset in [("validation", val_data), ("holdout", holdout_data)]:
                if len(subset) == 0:
                    continue
                fired  = subset["anomaly_score_weighted"] >= threshold
                result = _eval_threshold(subset, fired, subset["future_return_20d"].mean())
                if result is None:
                    continue
                rows.append({
                    "detector": "anomaly_score_weighted",
                    "signal":   f"anomaly_score_weighted >= {threshold}",
                    "split":    split_name,
                    "threshold": threshold,
                    **result,
                })

    # ── LSTM-AE: ae_error percentile buckets ─────────────────────────────────
    # Percentile thresholds computed on validation only (no lookahead into holdout)
    if "ae_error" in labeled.columns and len(val_data) > 0:
        for pct in [95, 90, 80]:
            threshold_val = val_data["ae_error"].quantile(pct / 100)  # from validation only
            for split_name, subset in [("validation", val_data), ("holdout", holdout_data)]:
                if len(subset) == 0:
                    continue
                fired  = subset["ae_error"] >= threshold_val
                result = _eval_threshold(subset, fired, subset["future_return_20d"].mean())
                if result is None:
                    continue
                rows.append({
                    "detector":  "lstm_ae",
                    "signal":    f"ae_error >= p{pct}  (>= {threshold_val:.4f})",
                    "split":     split_name,
                    "threshold": threshold_val,
                    **result,
                })

    result_df = pd.DataFrame(rows).sort_values(["detector", "split", "precision"], ascending=[True, True, False])
    result_df.to_parquet(OUT_DIR / "anomaly_precision.parquet", index=False)
    print(f"  Saved: {OUT_DIR / 'anomaly_precision.parquet'} ({len(result_df)} rows)")

    # Print summary: best threshold (by validation F1) and its holdout performance
    if not result_df.empty and "anomaly_score_weighted" in result_df["detector"].values:
        val_rows = result_df[(result_df["detector"] == "anomaly_score_weighted") & (result_df["split"] == "validation")]
        if not val_rows.empty:
            best = val_rows.loc[val_rows["f1"].idxmax()]
            best_t = best["threshold"]
            ho_row = result_df[
                (result_df["detector"] == "anomaly_score_weighted") &
                (result_df["split"] == "holdout") &
                (result_df["threshold"] == best_t)
            ]
            print(f"  Best threshold (validation F1): {best_t}  |  val_prec={best['precision']:.1%}  val_f1={best['f1']:.3f}")
            if not ho_row.empty:
                h = ho_row.iloc[0]
                print(f"  Holdout result for same threshold: prec={h['precision']:.1%}  recall={h['recall']:.1%}  f1={h['f1']:.3f}")

    return result_df


def print_summary(results: pd.DataFrame, precision_df: pd.DataFrame):
    total = len(results)
    tp = results["true_positive"].sum()
    fp = results["false_positive"].sum()
    tn = results["true_negative"].sum()
    fn = results["false_negative"].sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    lines = [
        "=" * 60,
        "BACKTEST RESULTS — FinWatch AI Decision Engine",
        "=" * 60,
        f"Total decisions evaluated : {total}",
        f"Walk-forward windows      : {results['test_start'].nunique()}",
        f"",
        f"OVERALL (CRITICAL + WARNING vs no-event):",
        f"  Precision : {precision:.1%}  (of alerts, how many had real drawdown)",
        f"  Recall    : {recall:.1%}  (of real drawdowns, how many were caught)",
        f"  F1 Score  : {f1:.3f}",
        f"  True  Positives: {tp}  |  False Positives: {fp}",
        f"  False Negatives: {fn} |  True  Negatives: {tn}",
        f"",
        f"SIGNAL PRECISION TABLE:",
        precision_df[["signal","n_fired","precision","recall","f1","avg_return_after_signal","return_edge"]].to_string(index=False),
        "",
        "=" * 60,
    ]
    report = "\n".join(lines)
    print(report)
    (OUT_DIR / "summary.txt").write_text(report)


def run():
    print("Loading detection data...")
    data = load_all_detection()
    print(f"  {len(data):,} rows, {data['ticker'].nunique()} tickers")
    print(f"  Date range: {data['Date'].min().date()} → {data['Date'].max().date()}")

    print("Adding stock-specific MA features...")
    from prediction.features.technical_signals import add_stock_ma_features
    data = add_stock_ma_features(data)

    print("Adding VIX context...")
    data = add_vix(data)

    print("Computing future outcomes...")
    data = compute_outcomes(data)

    print(f"\nRunning walk-forward backtest (train={TRAIN_YEARS}y, step={STEP_MONTHS}mo)...")
    results = run_walk_forward(data)

    if results.empty:
        print("No results generated.")
        return

    results.to_parquet(OUT_DIR / "backtest_results.parquet", index=False)
    print(f"\nSaved: {OUT_DIR / 'backtest_results.parquet'} ({len(results)} rows)")

    print("\nComputing signal precision...")
    precision_df = compute_signal_precision(results)
    precision_df.to_parquet(OUT_DIR / "signal_precision.parquet", index=False)
    print(f"Saved: {OUT_DIR / 'signal_precision.parquet'}")

    print("\nComputing anomaly detector precision (Option A)...")
    compute_anomaly_precision(data)

    print()
    print_summary(results, precision_df)


if __name__ == "__main__":
    run()
