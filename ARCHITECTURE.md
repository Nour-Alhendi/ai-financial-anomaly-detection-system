# FinWatch AI
––––––––––––––

Product Name:    FinWatch AI
GitHub Repo:     finwatch-ai
Description:     AI-Driven Financial Anomaly Detection
                 and Monitoring System



# LAYER 1 – DATA INGESTION
├── Stooq API (historical, 10 years, daily)
├── 55 Stocks, OHLCV (Tech, Finance, Health, Energy, Industrials, Green Energy)
├── 10 Reference ETFs (SPX, XLK, XLF, XLV, XLP, XLE, XLY, XLI, BOTZ, ICLN)
└── Output: raw_clean parquet files (data/raw/raw_clean/)

# LAYER 2 – DATA QUALITY CHECKS
├── Schema validation
├── Missing values
├── Duplicates
├── OHLC violations
└── Output: validated parquet files

# LAYER 3 – FEATURE ENGINEERING
│
├── 3A: Basic Features
│   ├── returns, volatility
│   ├── rolling_mean, rolling_std (20d)
│   ├── beta, corr_spx
│   ├── rsi
│   └── max_drawdown_30d (rolling 30-day peak-to-trough, no lookahead)
│
├── 3B: Context Features
│   ├── spx_return, etf_return
│   ├── is_market_wide, is_sector_wide   (pure market signal, no anomaly)
│   ├── excess_return, relative_return, sector_relative
│   ├── regime (bull/bear/transition), ma50, ma200, regime_encoded
│   └── vol_regime, volume_ma20, volume_zscore, is_high_volume
│
└── 3C: Advanced Features
    ├── return_lag_1/2/3, momentum_5/10
    ├── vol_5, vol_20, vol_change
    ├── trend_strength
    ├── volatility_ratio (vs SPX)
    └── volume_trend

# LAYER 4 – ANOMALY DETECTION
│
├── Statistical:    Z-Score (±3σ, 20d + 60d window)
├── ML:             Isolation Forest (group-aware, percentile threshold per group)
├── Deep Learning:  Dual LSTM Autoencoder
│   ├── low_vol_model  → trained on calm periods per group
│   ├── high_vol_model → trained on volatile periods per group
│   └── regime-specific thresholds from train data only (no lookahead)
├── Combine:
│   ├── anomaly_score (0–4): z + z_60 + if + ae
│   ├── combined_anomaly (bool)
│   ├── market_anomaly  = is_market_wide  & combined_anomaly
│   └── sector_anomaly  = is_sector_wide  & combined_anomaly
├── Severity:       normal / watch / warning / critical
└── Output: data/detection/ parquet files

# LAYER 5 – RISK & DIRECTION PREDICTION
│
├── 5A: Prediction Features (src/prediction/features/)
│   ├── Expected Shortfall (expected_shortfall.py):
│   │   ├── Rolling 252-day VaR (95%) + ES per ticker
│   │   ├── ES captures tail risk — not the same as volatility:
│   │   │   └── same volatility, very different ES → different danger level
│   │   └── Columns: var_95, es_95, es_ratio (ES/VaR)
│   ├── OBV Signal (obv_signal.py):
│   │   ├── obv_signal = volume_zscore × returns
│   │   ├── positive → high volume + positive return = institutional buying
│   │   ├── negative → high volume + negative return = panic selling
│   │   └── Used in: Decision Layer (sanity check) + Narrative Engine (confirmation)
│   └── Output: written to detection parquets + data/risk/ snapshot
│
├── 5B: XGBoost Risk Classifier (src/prediction/models/xgboost_risk.py)
│   ├── 29 features: anomaly signals, price/returns, context, volume, ES (var_95, es_95, es_ratio)
│   ├── Labels (forward-looking, 5 days) — VaR-based, per ticker, no lookahead:
│   │   ├── low  → max drawdown < 85th percentile
│   │   └── high → max drawdown ≥ 85th percentile
│   ├── Class imbalance handled via scale_pos_weight
│   └── Output: risk_level + p_low, p_high per ticker
│
├── 5C: XGBoost Direction Classifier (src/prediction/models/xgboost_direction.py)
│   ├── 29 features: same as 5B
│   ├── Labels (forward-looking, 5 days) — percentile-based, per ticker, no lookahead:
│   │   ├── up     → future_return ≥ 75th percentile
│   │   ├── down   → future_return ≤ 25th percentile
│   │   └── stable → in between
│   ├── Class imbalance handled via sample_weight="balanced"
│   └── Output: direction + p_up, p_stable, p_down per ticker
│
└── Output: models/xgboost_risk.pkl + models/xgboost_direction.pkl

# LAYER 6 – GUARDRAILS + DECISION ENGINE
├── Input: risk_level (high/low) × direction (up/stable/down) × anomaly signals × ES ratio × drawdown
│
├── Priority order:
│   ├── 1. Hard Override    — max_drawdown_30d ≤ -5% → always CRITICAL
│   ├── 2. REVIEW           — anomaly_score=0 + risk=high (models contradict)
│   ├── 3. ES Ratio Override — risk=high + es_ratio ≥ 1.5 → WARNING (regardless of model confidence)
│   ├── 4. Core Matrix      — risk=high + p_high ≥ 0.60:
│   │   ├── high + down (confirmed)   → CRITICAL
│   │   ├── high + down (weak)        → WARNING
│   │   ├── high + stable             → WARNING
│   │   └── high + up                 → WATCH (Dead Cat Bounce flag)
│   ├── 5. Confidence Catch — risk=high + p_high < 0.60 → WATCH (no silent fallthrough)
│   ├── 6. Low Risk:
│   │   ├── low + up + rising momentum → POSITIVE_MOMENTUM
│   │   └── low + down (confirmed)     → WATCH
│   └── 7. Default          → NORMAL
│
├── Severity levels: CRITICAL / WARNING / WATCH / NORMAL / POSITIVE_MOMENTUM / REVIEW
├── Actions: ESCALATE / MONITOR / OBSERVE / NONE / FLAG
└── Output: final decision + severity + confidence + context + summary per ticker

# LAYER 7 – XAI + NARRATIVE ENGINE + REPORTING
│
├── 7A: XAI — SHAP (src/explainability/xai.py)
│   ├── TreeExplainer on XGBoost Risk model
│   ├── Top-3 SHAP drivers per ticker (excludes context features)
│   └── Output: data/explanations/explanations.parquet
│
├── 7B: Narrative Engine (src/explainability/narrative_engine.py)
│   ├── Inputs: Decision (Layer 6) + SHAP Top-3 + OBV Signal
│   ├── driver       = Top-1 SHAP Feature (primary risk driver)
│   ├── confirmation = OBV direction (sell_pressure / buy_pressure / neutral)
│   ├── conflict     = obv_contradicts_severity | obv_signals_hidden_risk | None
│   └── narrative patterns:
│       ├── signals_aligned_bearish  — CRITICAL + OBV negative + driver↑risk
│       ├── signals_aligned_bullish  — NORMAL/WATCH + OBV positive + driver↓risk
│       ├── conflict_dead_cat_bounce — CRITICAL/WARNING + OBV strongly positive
│       ├── blind_spot_review        — NORMAL/WATCH + OBV strongly negative
│       └── mixed                   — no dominant pattern
│
├── 7C: LLM Narrator (src/explainability/llm_narrator.py)
│   ├── Input: Narrative Engine output (structured)
│   └── Output: Plain-English summary per ticker
│
├── 7D: FinBERT (src/explainability/finbert.py)
│   ├── Financial sentiment analysis on news headlines
│   └── Output: sentiment scores per ticker (positive / neutral / negative)
│
└── 7E: Report (src/explainability/report.py)
    ├── Per ticker: Severity + SHAP Top-3 + OBV + LLM Summary + News Sentiment
    └── Output: data/explanations/llm_news_enriched.parquet

# LAYER 8 – LOGGING & AUDIT (src/reporting/)
│
├── 8A: Anomaly Log (anomaly_log.py)
│   ├── Appends every pipeline run with UUID + timestamp
│   ├── Columns: run_id, timestamp, ticker, date, severity, action, confidence, context
│   └── Output: data/logs/anomaly_log.parquet
│
└── 8B: Daily Report (daily_report.py)
    ├── Management-level summary per run
    ├── Severity breakdown + ESCALATE / MONITOR / POSITIVE_MOMENTUM tiers
    └── Output: data/reports/daily_summary.txt

# STRESS TESTING (src/stress_testing/)
├── 11 data corruption scenarios (missing values, price spikes, OHLC violations, etc.)
├── Injection rates per scenario (0.3% – 2%)
├── Input:  data/raw/raw_clean/
└── Output: data/raw/raw_corrupted/ + data_quality_alert column
