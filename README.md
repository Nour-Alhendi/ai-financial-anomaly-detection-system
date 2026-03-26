## Project Status

This project is currently a **Work in Progress (WIP)**.
Submission deadline: March 30, 2026 — WBS Coding School Final Project.

---

## Overview

An AI-powered monitoring system for financial time-series data that automatically detects market anomalies, classifies their risk level, and explains why they happened.

The system moves beyond simple anomaly detection — it understands **context**, makes **predictions**, and supports **data-driven decisions**.

---

## Problem Statement

In risk management, you always want to have a plan and know when anomalies could occur — and know how to deal with them quickly.

You need to understand:
- Why did this happen?
- Was it expected?
- How dangerous is it?
- What decision should I make?

Traditional anomaly detection systems only flag unusual values but do not provide context or decision support. This project addresses that gap.

---

## System Architecture

The system follows an 8-layer modular pipeline:

| Layer | Name | Description |
|-------|------|-------------|
| 1 | Data Ingestion | Stooq API, 55 stocks, 10 years daily OHLCV |

| 2 | Data Quality Checks | Missing values, duplicates, gaps, schema validation |

| 3 | Feature Engineering | Basic (returns, volatility, RSI) + Context (regime, ETFs) + Advanced (momentum, lags) |

| 4 | Anomaly Detection | Z-Score + Isolation Forest + Dual LSTM Autoencoder — combined score + severity |

| 5 | Risk & Direction Prediction | XGBoost Risk (VaR-based, high/low) + XGBoost Direction (up/stable/down) |

| 6 | Guardrails + Decision Engine | Rule-based priority matrix: Risk × Direction → Severity + Action |

| 7 | XAI + Narrative Engine + Reporting | SHAP, Narrative Engine, LLM summaries, FinBERT news sentiment |

| 8 | Logging & Audit | Persistent audit log + daily management summary |

---

## Data Sources

- **Stooq** — historical daily OHLCV data (primary source)
- **55 stocks** across 9 sectors (Technology, AI & Robotics, Financials, Healthcare, Consumer Staples, Energy, Consumer Discretionary, Industrials, Green Energy)
- **10 Sector ETFs** — XLK, BOTZ, XLF, XLV, XLP, XLE, XLY, XLI, ICLN, ^SPX
- **Reference Index** — ^SPX (S&P 500)

---

## Key Features

- Multi-asset monitoring (55 stocks, 10 sector ETFs)
- Three-layer anomaly detection (Statistical + ML + Deep Learning)
- Severity classification (CRITICAL / WARNING / WATCH / NORMAL / POSITIVE_MOMENTUM / REVIEW)
- Market context (market-wide vs sector-specific vs stock-specific anomaly)
- Market regime detection (Bull / Bear / Transition)
- Risk classification with XGBoost (VaR-based labels, high/low)
- Direction prediction with XGBoost (up/stable/down, 5-day horizon)
- Expected Shortfall (VaR 95% + ES tail risk per ticker)
- SHAP-based explainability (top-3 drivers per ticker)
- Narrative Engine + LLM summaries in plain English
- FinBERT news sentiment analysis
- Streamlit dashboard (FinWatch AI)
- Stress testing framework (11 data corruption scenarios)
- Full audit logging + daily management report

---

## What This Project Is NOT

- Not a trading bot
- Not an investment advisory system
- Not a high-frequency trading engine

It is a **monitoring and decision-support framework**.

---

## Project Structure

```
ai-monitoring-system/
├── config/
│   └── assets.yaml              # Stock tickers, sectors, ETF mappings
├── data/
│   ├── raw/                     # Downloaded OHLCV data
│   ├── features/                # Engineered features
│   ├── detection/               # Anomaly detection results
│   ├── decisions/               # Decision engine output
│   ├── explanations/            # SHAP + LLM summaries
│   ├── logs/                    # Audit trail
│   ├── reports/                 # Daily management summaries
│   └── risk/                    # Expected Shortfall snapshots
├── src/
│   ├── ingestion/               # Layer 1: Data download
│   ├── quality/                 # Layer 2: Data quality checks
│   ├── features/                # Layer 3: Feature engineering
│   │   ├── basic/               # Returns, volatility, RSI, beta, drawdown
│   │   ├── context/             # Regime, ETFs, excess return
│   │   └── advanced/            # Momentum, lags, vol_change, trend
│   ├── detection/               # Layer 4: Anomaly detection
│   ├── prediction/              # Layer 5: Risk & Direction prediction
│   │   ├── features/            # ES + OBV signal computation
│   │   └── models/              # XGBoost risk + direction classifiers
│   ├── decision/                # Layer 6: Decision engine
│   ├── explainability/          # Layer 7: SHAP, Narrative Engine, LLM, FinBERT
│   ├── reporting/               # Layer 8: Audit log + daily report
│   ├── intelligence/            # RAG knowledge base (retriever + narrator)
│   └── pipeline.py              # Main entry point (end-to-end)
├── app.py                       # Streamlit dashboard (FinWatch AI)
├── models/                      # Trained model files (.pkl, .keras)
├── knowledge_base/              # Documents for RAG retrieval
└── ARCHITECTURE.md              # Detailed architecture documentation
```

---

## Future Improvements

- Real-time / intraday data (yfinance 1h)
- Performance benchmarking against buy-and-hold baseline
- Expanded asset universe (international markets, crypto)
- Alert delivery via email / Slack
