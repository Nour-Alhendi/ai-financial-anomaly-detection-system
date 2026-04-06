# FinWatch AI — System Architecture

**AI-Driven Financial Anomaly Detection and Decision Support**

This document covers the full technical design of the 8-layer pipeline — model choices, feature engineering, decision logic, and the reasoning behind each design decision.

---

## System Overview

```
Raw OHLCV
    │
    ▼
[ Layer 1 ] Data Ingestion         → data/raw/{ticker}.parquet
    │
    ▼
[ Layer 2 ] Data Quality           → validated parquet
    │
    ▼
[ Layer 3 ] Feature Engineering    → data/processed/{ticker}.parquet
    │
    ▼
[ Layer 4 ] Anomaly Detection      → data/detection/{ticker}.parquet
    │
    ▼
[ Layer 5 ] Prediction + Fundamentals → data/predictions/ + data/fundamental/
    │
    ▼
[ Layer 6 ] Decision Engine        → data/decisions/decisions.parquet
    │
    ▼
[ Layer 7 ] Explainability + Sentiment → data/explanations/
    │
    ▼
[ Layer 8 ] Reporting + Dashboard  → data/logs/ + data/reports/ + Streamlit
```

---

## Layer 1 — Data Ingestion

- **Source:** Stooq API (free, no rate limits for historical pulls)
- **Universe:** 55 stocks across 9 sectors + 10 sector ETFs + S&P 500 reference (`^SPX`)
- **History:** 10 years daily OHLCV
- **Format:** One Parquet file per ticker in `data/raw/`

**Sectors covered:** Technology, AI & Robotics, Financials, Healthcare, Consumer Staples, Energy, Consumer Discretionary, Industrials, Green Energy, Crypto

**ETF reference set:** XLK, XLF, XLV, XLP, XLE, XLY, XLI, BOTZ, ICLN, ^SPX  
Used as sector-level baselines for excess return and regime detection.

---

## Layer 2 — Data Quality

Five validation checks run before any feature is computed:

| Check | What it catches |
|-------|----------------|
| Schema validation | Wrong column types, missing columns |
| Missing values | NaN rows, -999 placeholders, zero-price rows |
| Duplicates | Exact row copies or date conflicts |
| OHLC violations | `High < Low`, `Open > High`, `Close < Low` |
| Gap detection | Missing trading days (beyond expected market holidays) |

**Design note:** Quality runs before feature engineering. Corrupt data reaching the model layer is harder to catch and produces silent errors — catching it upstream prevents silent failures downstream. See [Stress Testing](#stress-testing) for the full validation coverage.

---

## Layer 3 — Feature Engineering

### 3A · Basic Features (`src/features/basic/`)

| Feature | Description |
|---------|-------------|
| `returns` | Daily log returns |
| `volatility` | 20-day rolling standard deviation of returns |
| `rolling_mean`, `rolling_std` | 20-day window |
| `beta` | Rolling beta vs. S&P 500 |
| `corr_spx` | Rolling 60-day correlation with S&P 500 |
| `rsi` | 14-day Relative Strength Index |
| `max_drawdown_30d` | Rolling 30-day peak-to-trough — no lookahead |

### 3B · Context Features (`src/features/context/`)

| Feature | Description |
|---------|-------------|
| `spx_return`, `etf_return` | Same-day market and sector reference returns |
| `excess_return` | Stock return − S&P 500 return |
| `relative_return` | Stock return − sector ETF return |
| `sector_relative` | Excess vs. sector median |
| `is_market_wide` | True if SPX also anomalous on same day |
| `is_sector_wide` | True if sector ETF also anomalous on same day |
| `regime` | Bull / Bear / Transition — based on MA50/MA200 crossover |
| `regime_encoded` | Integer encoding for model input |
| `vol_regime` | High / normal / low volatility regime |
| `volume_ma20`, `volume_zscore`, `is_high_volume` | Volume relative to 20d baseline |

**Why regime matters:** A `WARNING` signal in a bull market has different implications than the same signal in a bear market. The decision engine uses regime to adjust thresholds and block certain signals.

### 3C · Advanced Features (`src/features/advanced/`)

| Feature | Description |
|---------|-------------|
| `return_lag_1/2/3` | Lagged returns for autocorrelation signal |
| `momentum_5`, `momentum_10` | 5- and 10-day price momentum |
| `vol_5`, `vol_20`, `vol_change` | Short vs. medium volatility |
| `trend_strength` | ADX-style trend magnitude |
| `volume_trend` | Volume acceleration vs. baseline |
| `volatility_ratio` | Stock volatility / SPX volatility |

---

## Layer 4 — Anomaly Detection

### Design: Why 4 Models?

No single model covers all anomaly types:
- Z-Score catches statistical return outliers but ignores multivariate structure
- Isolation Forest captures multivariate outliers but ignores temporal sequence
- LSTM Autoencoder captures temporal dependencies but requires per-sector training

Using four complementary detectors and combining them into a weighted score produces a more robust signal than any single model.

### Model Details

| Model | File | Instances | Weight | What it detects |
|-------|------|-----------|--------|----------------|
| LSTM Autoencoder | `src/detection/lstm_autoencoder.py` | 30 (sector × volatility bucket) | 0.30 | Sequence anomalies — reconstruction error on 20-day windows |
| Isolation Forest | `src/detection/isolation_forest.py` | 14 (one per sector) | 0.30 | Multivariate outliers in feature space |
| Return Z-Score | `src/detection/statistical.py` | — | 0.20 | Distribution outliers (±3σ, 20d and 60d window) |
| Sector Z-Score | `src/detection/statistical.py` | — | 0.20 | Stock return vs. sector peer distribution |

### Output Fields

| Field | Description |
|-------|-------------|
| `anomaly_score_weighted` | Weighted sum, continuous 0–1 |
| `anomaly_score` | Integer 0–4 (legacy, kept for compatibility) |
| `combined_anomaly` | Boolean — score above calibrated threshold |
| `market_anomaly` | `is_market_wide` AND `combined_anomaly` |
| `sector_anomaly` | `is_sector_wide` AND `combined_anomaly` |

**Threshold calibration:** The anomaly threshold is selected on the validation set and evaluated on the holdout (2024+) — never tuned on holdout data. Calibration runs as part of `backtest.py`.

**Output:** `data/detection/{ticker}.parquet`

---

## Layer 5 — Prediction + Fundamental Signals

### 5A · Tail Risk Features (`src/prediction/features/`)

**Expected Shortfall (`expected_shortfall.py`)**
- Rolling 252-day historical VaR at 95% confidence
- `var_95` — Value at Risk threshold
- `es_95` — Expected Shortfall (average loss beyond VaR)
- `es_ratio` — ES / VaR (tail severity ratio)

**OBV Signal (`obv_signal.py`)**
- `obv_signal = volume_zscore × returns`
- Positive → volume-backed price increase (institutional buying)
- Negative → high-volume selloff (panic selling / distribution)

### 5B · Drawdown Probability Model (`src/prediction/models/drawdown_probability.py`)

- **Type:** XGBoost + LightGBM (best model selected automatically via marker file)
- **Target:** P(max drawdown > 5% over next 10 trading days)
- **Train/Holdout split:** All data before 2024 for training; 2024–2026 as unseen holdout
- **AUC on holdout:** 0.715
- **Features:** 61 total — anomaly signals, price/returns, momentum, MA positions, volume, ES, VIX, 15 technical signals (Golden Cross, Death Cross, MACD, RSI divergence, panic volume, etc.)
- **Model file:** `models/xgboost_drawdown.pkl` or `models/lgbm_drawdown.pkl` (winner written to `models/best_drawdown_model.txt`)

**Why XGBoost + LightGBM?**  
Both are trained on every run. The one with higher test AUC (by margin > 0.005) wins and is used in production. This gives a free ensemble comparison without extra cost. 10-day horizon was chosen over 20-day because shorter horizons are more predictable — AUC improved from 0.674 → 0.715.

### 5C · Meta-Model Stacking (`src/prediction/models/meta_model.py`)

- **Type:** Logistic Regression (L2 regularized)
- **Inputs:** `p_drawdown` + anomaly score + VIX regime
- **Output:** `p_drawdown_meta` — refined probability after stacking
- **Trained on:** Walk-forward backtest fold outputs
- **Model file:** `models/meta_model.pkl`

**Why stacking?**  
The XGBoost model is trained on all historical data. The meta-model is trained specifically on the distribution of backtest predictions — it learns to correct for systematic over- or under-confidence in different market regimes.

### 5D · Fundamental Collectors (`src/ingestion/`)

All collectors write to `data/fundamental/` with a 1-day lag by design (data is available the next trading day):

| Collector | Output file | Signal |
|-----------|-------------|--------|
| Valuation | `valuation.parquet` | `pe_ratio`, `pe_forward`, `pb_ratio`, `revenue_growth` |
| Insider | `insider.parquet` | `insider_sentiment` (−1 to +1) |
| Options | `options.parquet` | `put_call_ratio`, `options_fear` |
| Earnings | `earnings.parquet` | `days_to_next_earnings` |
| Sentiment | `sentiment.parquet` | Historical headlines (input to Layer 7) |

**Data source:** yfinance (fundamentals), Finnhub (headlines)

---

## Layer 6 — Decision Engine

**Files:** `src/decision/decision_engine.py` + `src/decision/decision_pipeline.py`

### Severity Classification (priority-ordered)

| Priority | Condition | Severity |
|----------|-----------|----------|
| 1 | `p_drawdown ≥ p_crit` AND `anomaly_w ≥ 0.30` | **CRITICAL** |
| 2 | Actual 30d drawdown ≤ −15% | **CRITICAL** |
| 3 | `p_drawdown ≥ p_warn` OR `anomaly_w ≥ 0.30` | **WARNING** |
| 4 | Actual drawdown ≤ −8% | **WARNING** |
| 5 | `anomaly_w ≥ 0.20` OR `p_drawdown ≥ p_watch` | **WATCH** |
| 6 | Bear regime | Upgrades WATCH → WARNING; blocks ENTRY |
| 7 | RSI > 75 + >15% above MA200 + positive momentum | **EXIT** (overbought profit-taking) |
| 8 | `p_dd < 30%` + RSI < 70 + positive momentum + `anomaly_w < 0.20` | **POSITIVE_SIGNAL** |
| 9 | High anomaly + low p_drawdown + outperforming | **REVIEW** (conflicting signals) |
| 10 | Default | **NORMAL** |

**VIX-aware thresholds:**  
At low VIX (< 20), `p_crit`, `p_warn`, and `p_watch` thresholds are raised. In calm markets, the model's drawdown probability predictions tend to overfire — raising thresholds suppresses false positives without retraining.

### Valuation Gates

| Condition | Effect |
|-----------|--------|
| `pe_ratio < 0` | Block ENTRY (negative earnings) |
| `pe_ratio > 50` | Block ENTRY (overvalued) |
| `pe_ratio < 15` OR `pb_ratio < 1.5` | Strengthen POSITIVE_SIGNAL ("cheap" note added) |
| `revenue_growth < −0.05` | Add "revenue declining" to context |

### Trading Signal Logic

**CRITICAL / WARNING → EXIT or HOLD** (momentum-recovery aware):

```
recovering    = momentum_5 > 0.03      # stock actively bouncing back
strong_signal = p_dd >= 0.50 or anomaly_w >= 0.35

if strong_signal and not recovering → EXIT
if strong_signal and recovering     → HOLD + "recovering momentum" note
if not strong_signal                → HOLD
```

**Design note:** Initial implementation triggered EXIT on all WARNING severity stocks. Testing revealed that 46/65 assets showed EXIT including stocks that had already recovered. The two-condition gate (strong ML signal AND no active recovery) reduced EXIT to 34 — qualitatively more accurate signals.

| Severity | Trading Signal |
|----------|---------------|
| CRITICAL | EXIT |
| WARNING | EXIT (if strong + not recovering) or HOLD |
| WATCH | HOLD |
| POSITIVE_SIGNAL | ENTRY (if above MA200 + bull/transition regime + `pe < 35` + revenue OK) or HOLD |
| NORMAL | HOLD |
| REVIEW | NEUTRAL |

### Output Fields

| Field | Values |
|-------|--------|
| `severity` | CRITICAL / WARNING / WATCH / POSITIVE_SIGNAL / NORMAL / REVIEW |
| `action` | ESCALATE / MONITOR / OBSERVE / NONE |
| `trading_signal` | ENTRY / HOLD / EXIT / NEUTRAL |
| `confidence` | Derived from backtest signal precision |
| `context` | Plain-text reasoning string (passed to LLM narrator) |

**Output:** `data/decisions/decisions.parquet`

---

## Layer 7 — Explainability + Sentiment

### 7A · SHAP Explainability (`src/explainability/xai.py`)

- **Method:** TreeExplainer on XGBoost Drawdown model
- **Output:** Top-3 SHAP drivers per ticker (context/reference features excluded from display)
- **Output file:** `data/explanations/explanations.parquet`

### 7B · News Sentiment (`src/explainability/finbert.py`)

- **Source:** Finnhub API — last 7 days of headlines per ticker
- **VADER score:** Lexicon-based, fast, no API cost, handles financial vocabulary reasonably
- **Groq score:** `llama-3.3-70b-versatile` — contextual financial analyst framing
- **Combined:** `news_sentiment_score = 0.4 × VADER + 0.6 × Groq`
- **Graceful degradation:** Falls back to VADER-only when Groq daily quota (100k TPD) is exhausted

**Why VADER + LLM instead of FinBERT?**  
FinBERT requires GPU for reasonable throughput and its pre-training corpus (financial filings) doesn't align well with short news headlines. VADER is fast and surprisingly competitive on headline-length text. The Groq LLM provides contextual framing that neither model can match for free-text analysis.

### 7C · LLM Narrator (`src/explainability/llm_narrator.py`)

- **Input:** Decision output + top-3 SHAP drivers + sentiment scores
- **Output:** Plain-English per-ticker narrative (grounded, hallucination-resistant)
- **Model:** `llama-3.3-70b-versatile` via Groq API
- **Caching:** JSON cache keyed by `YYYY-MM-DD_TICKER` — prevents re-generation within the same trading day and avoids quota exhaustion on re-runs

**Hallucination control:** The LLM is prompted with structured facts only (severity, signal, SHAP features, sentiment score, context string). It synthesizes language — it does not generate new financial claims.

### 7D · Explainability Pipeline (`src/explainability/explainability_pipeline.py`)

Orchestrates 7A → 7B → 7C in sequence.  
**Output:** `data/explanations/llm_news_enriched.parquet`

---

## Layer 8 — Reporting + Dashboard

### 8A · Audit Log (`src/reporting/anomaly_log.py`)

Append-only log per pipeline run:
- Fields: `run_id`, `timestamp`, `ticker`, `severity`, `action`, `confidence`
- **Output:** `data/logs/anomaly_log.parquet`

### 8B · Daily Report (`src/reporting/daily_report.py`)

Management-level text summary:
- Severity breakdown across all assets
- ESCALATE tier (CRITICAL + WARNING)
- MONITOR tier (WATCH)
- **Output:** `data/reports/daily_summary.txt`

### 8C · Dashboard (`finwatch/app.py`)

Streamlit, dark theme, 4 pages:
- **Landing** — portfolio-level severity heatmap
- **Stock view** — severity, trading signal, price chart, RSI, LLM narrative, SHAP drivers, news sentiment
- **Portfolio** — cross-asset breakdown
- **Sector ETF** — reference ETF overview

**AI Analyst Report:** PDF export — multilingual (EN / DE / AR via Groq translation)

---

## Backtesting (`src/backtesting/backtest.py`)

Walk-forward setup — no lookahead bias:

- **Train window:** 4-year rolling
- **Test windows:** 6-month steps

| Signal | n | Avg 20d Return | Drawdown Rate |
|--------|---|----------------|---------------|
| ENTRY | 12 | +3.37% | 17% |
| HOLD | 50 | +2.94% | 26% |
| EXIT | 432 | +3.79% | 36% |

**Interpretation:** Risk ordering is preserved — EXIT flags appear on assets that subsequently suffer the highest drawdown rate. ENTRY flags are on assets with the lowest drawdown risk. This is the correct behavior for a risk-first monitoring system.

**Outputs:** `data/backtesting/backtest_results.parquet`, `signal_precision.parquet`, `anomaly_precision.parquet`, `summary.txt`

---

## Data Flow

```
data/raw/{ticker}.parquet              ← Layer 1: download_historical.py
data/processed/{ticker}.parquet        ← Layer 3: feature_pipeline.py
data/detection/{ticker}.parquet        ← Layer 4: detection_pipeline.py
data/predictions/{ticker}.parquet      ← Layer 5: prediction_pipeline.py
data/fundamental/*.parquet             ← Layer 5: fundamental collectors (1-day lag)
data/decisions/decisions.parquet       ← Layer 6: decision_pipeline.py
data/explanations/explanations.parquet ← Layer 7A: xai.py
data/explanations/llm_news_enriched.parquet ← Layer 7D: explainability_pipeline.py
data/backtesting/                      ← backtest.py
data/logs/anomaly_log.parquet          ← Layer 8A: anomaly_log.py
data/reports/daily_summary.txt         ← Layer 8B: daily_report.py
```

---

## Model Registry

| Model | File | Trained on | Used in |
|-------|------|-----------|---------|
| XGBoost Drawdown | `models/xgboost_drawdown.pkl` | Pre-2024 data | Layer 5B + Layer 7A (SHAP) |
| LightGBM Drawdown | `models/lgbm_drawdown.pkl` | Pre-2024 data | Layer 5B (comparison — best wins) |
| Meta-Model | `models/meta_model.pkl` | Backtest fold outputs | Layer 5C |
| Isolation Forest × 14 | `models/if_{sector}.pkl` | Per-sector history | Layer 4 |
| LSTM Autoencoder × 30 | `models/ae_{sector}_{bucket}.keras` | Per-sector × volatility | Layer 4 |
