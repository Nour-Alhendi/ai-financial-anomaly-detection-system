import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.download_historical         import run as run_download
from ingestion.sentiment_collector         import collect as collect_sentiment, save as save_sentiment
from ingestion.earnings_collector          import collect as collect_earnings, save as save_earnings
from ingestion.insider_collector           import collect as collect_insider, save as save_insider
from ingestion.options_collector           import collect as collect_options, save as save_options
from ingestion.valuation_collector         import collect as collect_valuation, save as save_valuation
from quality.quality_pipeline              import run_quality_pipeline
from features.feature_pipeline             import run_feature_pipeline
from detection.detection_pipeline          import run as run_detection
from prediction.prediction_pipeline        import run_prediction_pipeline
from decision.decision_pipeline            import run as run_decision
from explainability.explainability_pipeline import run as run_explainability
from reporting.anomaly_log                 import log as log_anomalies
from reporting.daily_report                import run as run_daily_report
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    # ── Step 0: Download latest price data
    print("\n" + "=" * 55)
    print("STEP 0 — Download latest data")
    print("=" * 55)
    run_download()

    # ── Steps 1–3: Data quality + features + detection
    run_quality_pipeline()
    run_feature_pipeline()
    run_detection()

    # ── Step 4: Collect fundamental signals (needed by Decision Engine)
    print("\n" + "=" * 55)
    print("STEP 4 — Collect fundamental signals")
    print("=" * 55)

    earnings_df = collect_earnings()
    if not earnings_df.empty:
        out = save_earnings(earnings_df)
        print(f"Earnings saved → {out}")

    insider_df = collect_insider()
    if not insider_df.empty:
        out = save_insider(insider_df)
        print(f"Insider saved → {out}")

    options_df = collect_options()
    if not options_df.empty:
        out = save_options(options_df)
        print(f"Options saved → {out}")

    valuation_df = collect_valuation()
    if not valuation_df.empty:
        out = save_valuation(valuation_df)
        print(f"Valuation saved → {out}")

    # ── Step 5: Prediction + Decision + Explainability
    run_prediction_pipeline()
    decisions_df = run_decision()
    run_explainability()
    log_anomalies(decisions_df)
    run_daily_report()

    # ── Step 6: Collect news sentiment history (for future training)
    print("\n" + "=" * 55)
    print("STEP 6 — Collect news sentiment history")
    print("=" * 55)
    sentiment_df = collect_sentiment()
    if not sentiment_df.empty:
        out = save_sentiment(sentiment_df)
        print(f"Sentiment saved → {out}")


