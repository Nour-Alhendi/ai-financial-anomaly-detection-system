# FinWatch AI

**AI-powered Financial Risk Assessment & Decision Support System**

> Monitors 55 stocks and 10 sector ETFs daily — detects anomalies, classifies risk severity, and produces an explainable trading signal for every asset.

---

## What It Does

Most anomaly detection systems stop at the flag: *"something is unusual here."*  
That is not enough to act on.

FinWatch AI goes further. It answers the question that actually matters after an anomaly is detected:

**Is the price still falling — or has it already bottomed out? Should I act now, or wait?**

The system produces a concrete, explainable recommendation (`ENTRY` / `HOLD` / `EXIT`) for every monitored asset, grounded in:

- Anomaly severity (4-model ensemble)
- Drawdown probability (XGBoost + LightGBM, AUC 0.715, 10-day horizon)
- Market regime (Bull / Bear / Transition)
- Valuation fundamentals (P/E, P/B, revenue growth)
- News sentiment (VADER + LLM analyst opinion)
- Momentum recovery signals

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data ingestion | Stooq API, yfinance, Finnhub |
| Storage | Parquet (per-ticker), structured `data/` layout |
| Feature engineering | pandas, numpy — 30+ features per ticker |
| Anomaly detection | LSTM Autoencoder (Keras), Isolation Forest (scikit-learn), Z-Score |
| Prediction | XGBoost, Logistic Regression meta-model stacking |
| Explainability | SHAP (TreeExplainer), VADER, Groq LLM (`llama-3.3-70b-versatile`) |
| Dashboard | Streamlit, dark theme, multilingual PDF export |
| Language | Python 3.11 |

---

## ML Architecture

The system runs an **8-layer modular pipeline**:

| Layer | Name | What it does |
|-------|------|-------------|
| 1 | Data Ingestion | Downloads 10 years of daily OHLCV for 65 assets |
| 2 | Data Quality | Validates schema, detects OHLC violations, gaps, stale prices |
| 3 | Feature Engineering | 30+ features: returns, RSI, momentum, regime, ETF context, sector comparison |
| 4 | Anomaly Detection | 4-model ensemble → weighted continuous score (0–1) |
| 5 | Prediction + Fundamentals | Drawdown probability + meta-model stacking + valuation signals |
| 6 | Decision Engine | Severity classification + trading signal, regime- and momentum-aware |
| 7 | Explainability + Sentiment | SHAP drivers + VADER + Groq LLM narrative |
| 8 | Reporting + Dashboard | Streamlit dashboard + audit log + daily management summary |

### Anomaly Detection — Ensemble of 4 Models

| Model | Count | Weight | What it detects |
|-------|-------|--------|----------------|
| LSTM Autoencoder | 30 (sector × volatility bucket) | 0.30 | Temporal sequence anomalies |
| Isolation Forest | 14 (per sector) | 0.30 | Multivariate outliers |
| Return Z-Score | — | 0.20 | Distribution outliers (±3σ, 20d + 60d window) |
| Sector Z-Score | — | 0.20 | Stock vs. sector peer deviation |

Combined into a single `anomaly_score_weighted` (0–1 continuous) per ticker per day.

### Prediction — XGBoost + Meta-Model Stacking

- **Target:** P(max drawdown > 5% over next 10 days)
- **Model:** XGBoost + LightGBM (best model selected automatically at training time)
- **AUC:** 0.715 on holdout set (2024–2026, unseen during training)
- **Meta-model:** Logistic Regression stacking — combines `p_drawdown` + anomaly signals + VIX into `p_drawdown_meta`

### Decision Engine — Severity + Trading Signal

**Severity classification** (priority-ordered rules):

| Priority | Condition | Severity |
|----------|-----------|----------|
| 1 | `p_drawdown ≥ threshold` AND `anomaly_score ≥ 0.30` | CRITICAL |
| 2 | Actual 30d drawdown ≤ −15% | CRITICAL |
| 3 | `p_drawdown` OR `anomaly_score ≥ 0.30` | WARNING |
| 4 | Actual drawdown ≤ −8% | WARNING |
| 5 | `anomaly_score ≥ 0.20` OR moderate `p_drawdown` | WATCH |
| 6 | `p_drawdown < 30%` + RSI < 70 + positive momentum | POSITIVE_SIGNAL |

VIX-aware: thresholds are raised at low VIX (< 20) to reduce false positives in calm markets.

**Trading signal** — momentum-recovery aware:

For `WARNING` severity, the system distinguishes between stocks still declining vs. those actively recovering:
- `momentum_5 > 0.03` (active recovery) + weak ML signal → `HOLD`
- Strong ML signal (`p_dd ≥ 0.50`) + no recovery → `EXIT`

Valuation gates: blocks `ENTRY` on negative or extreme P/E (> 50); strengthens `POSITIVE_SIGNAL` on cheap fundamentals.

---

## Backtesting Results

Walk-forward setup — no lookahead bias. 4-year rolling train window, 6-month test windows.

| Signal | Avg 20d Return | Drawdown Rate |
|--------|----------------|---------------|
| ENTRY  | +3.37%         | 17%           |
| HOLD   | +2.94%         | 26%           |
| EXIT   | +3.79%         | 36%           |

Risk ordering is correct: `EXIT` signals carry the highest drawdown rate, `ENTRY` the lowest. The system correctly identifies which situations are most dangerous.

---

## Key Design Decisions

**Why an ensemble of 4 anomaly detectors?**  
Each model has different blind spots. LSTM captures temporal patterns, Isolation Forest finds multivariate outliers, Z-Score catches distribution extremes. The ensemble is more robust than any single model.

**Why not just use the anomaly score to trigger EXIT?**  
An anomaly means something unusual happened — not that the price is still falling. A stock can spike down and immediately recover. The prediction layer (drawdown probability) adds the time dimension: *will this continue?*

**Why momentum-aware EXIT logic?**  
Early testing showed 46 stocks triggering EXIT even when prices had already recovered. By gating EXIT on both strong ML conviction (`p_dd ≥ 0.50`) and absence of recovery momentum, the signal count dropped from 46 to 34 — with qualitatively better signal quality.

**Why SHAP + LLM?**  
SHAP tells you *which features* drove the model decision. The LLM (Groq / llama-3.3-70b) translates that into plain English for non-technical stakeholders, grounded strictly on model outputs to avoid hallucination.

---

## Dashboard

Interactive Streamlit dashboard (dark theme) with:
- Per-stock view: severity badge, trading signal, price chart, RSI, LLM narrative, SHAP drivers
- Portfolio view: severity distribution across all assets
- Sector ETF overview
- AI Analyst Report: PDF export in English, German, and Arabic (via Groq translation)

---

## How to Run

**Train models (first time only):**

```bash
python src/prediction/models/drawdown_probability.py   # XGBoost + LightGBM drawdown model
python src/detection/isolation_forest.py               # Isolation Forest × 14 sectors
python src/detection/lstm_autoencoder.py               # LSTM Autoencoder × 30 buckets
python src/prediction/models/meta_model.py             # Meta-model stacking layer
python src/backtesting/backtest.py                     # Walk-forward backtest + signal precision
```

> Models are **not retrained** on every pipeline run — only predictions are made.
> Retrain recommendation: Drawdown model monthly, Isolation Forest every 3–6 months, LSTM Autoencoder every 6 months, Meta-model after each Drawdown retrain.

**Daily pipeline:**

```bash
python src/pipeline.py
```

**Dashboard:**

```bash
cd finwatch
streamlit run app.py
```

---

## Data Sources

| Source | What it provides |
|--------|-----------------|
| Stooq | Daily OHLCV — 10 years historical |
| Finnhub | News headlines — last 7 days per ticker |
| yfinance | Fundamentals — P/E, P/B, revenue growth, insider activity, options flow |

**Universe:** 55 stocks across 9 sectors + 10 sector ETFs + S&P 500 reference index.

---

## Project Structure

```
ai-monitoring-system/
├── config/assets.yaml               # Tickers, sectors, ETF mappings
├── src/
│   ├── ingestion/                   # Data download + fundamental collectors
│   ├── quality/                     # Data quality validation
│   ├── features/                    # Feature engineering (basic / context / advanced)
│   ├── detection/                   # Anomaly detection (LSTM-AE, IF, Z-Score)
│   ├── prediction/                  # Drawdown model, meta-model, ES, OBV
│   ├── decision/                    # Severity + trading signal logic
│   ├── explainability/              # SHAP, VADER, Groq narrator
│   ├── reporting/                   # Audit log, daily report
│   ├── backtesting/                 # Walk-forward backtesting
│   └── pipeline.py                  # Main entry point
├── finwatch/app.py                  # Streamlit dashboard
├── models/                          # Trained model files (.pkl, .keras)
└── ARCHITECTURE.md                  # Full technical architecture
```

---

## Scope

This is a **monitoring and decision-support framework** for human analysts — not a trading bot and not investment advice. It runs on daily OHLCV data and is designed to be the analytical layer before a human makes a decision.

---

## Planned

- AI Agent with RAG — natural language Q&A over portfolio state ("Why is TSLA flagged? What do analysts say?")
- Real-time / intraday data (1h candles)
- Alert delivery via email or Slack
- Expanded universe (international markets, crypto)
