"""
FinWatch AI — Layer 7C: LLM Narrator
======================================
Takes structured Narrative Engine output + detection + decision data
and produces a full professional analysis using Groq (Llama 3.3 70B).

Supported languages: english, german, arabic

Performance: by default only processes CRITICAL + WARNING tickers to stay
within Groq free tier limits (30 req/min). Use severity_filter=None for all.
"""

import os
import json
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
from dotenv import load_dotenv
from groq import Groq

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MODEL       = "llama-3.3-70b-versatile"
TEMPERATURE = 0.2
MAX_TOKENS  = 700   # reduced from 1100 — saves ~35% tokens per call

DAILY_TOKEN_BUDGET = 80_000   # stay safely under Groq free tier (100k/day)
_tokens_used = 0              # tracked across calls in this run

LANGUAGE_INSTRUCTION = {
    "english": "Respond in English.",
    "german":  "Antworte auf Deutsch.",
    "arabic":  "أجب باللغة العربية.",
}

DEFAULT_SEVERITY_FILTER = {"CRITICAL", "WARNING"}

CACHE_PATH = ROOT / "data/explanations/llm_cache.json"


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def _cache_key(ticker: str) -> str:
    return f"{date.today().isoformat()}_{ticker}"


def _load_detection_latest(ticker: str, detection_dir: Path) -> dict:
    """Load latest row + historical context (EMA20, monthly summaries, 3M range)."""
    f = detection_dir / f"{ticker}.parquet"
    if not f.exists():
        return {}
    df = pd.read_parquet(f)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    r = df.iloc[-1].to_dict()

    closes = df["Close"].values
    r["current_price"] = float(closes[-1])
    r["ema_20"] = float(pd.Series(closes).ewm(span=20).mean().iloc[-1])

    df_3m = df[df["Date"] >= df["Date"].max() - pd.Timedelta(days=90)]
    r["price_3m_high"] = float(df_3m["Close"].max())
    r["price_3m_low"]  = float(df_3m["Close"].min())

    monthly = []
    for label, grp in df.tail(120).groupby(df.tail(120)["Date"].dt.to_period("M")):
        monthly.append(f"{label}: high={grp['Close'].max():.2f} low={grp['Close'].min():.2f} close={grp['Close'].iloc[-1]:.2f}")
    r["monthly_summary"] = " | ".join(monthly[-4:])

    return r


def _build_prompt(row: dict, language: str) -> str:
    lang = LANGUAGE_INSTRUCTION.get(language, LANGUAGE_INSTRUCTION["english"])

    # Technical values
    rsi        = row.get("rsi", 50)
    mom5       = row.get("momentum_5", 0)
    mom10      = row.get("momentum_10", 0)
    vol        = row.get("volatility", 0)
    ret_1d     = row.get("returns", 0)
    drawdown   = row.get("max_drawdown_30d", 0)
    regime     = row.get("regime", "unknown")
    excess     = row.get("excess_return", 0)
    vol_ma20   = row.get("volume_ma20", 1) or 1
    volume     = row.get("Volume", 0)
    vol_ratio  = (volume / vol_ma20) if vol_ma20 > 0 else 1.0
    obv        = row.get("obv_signal", 0)
    confirm    = row.get("confirmation", "neutral")

    # Direction model — use real probabilities from decisions.parquet
    direction  = row.get("direction", "stable")
    p_down     = float(row.get("p_down", 0.33))
    p_up       = float(row.get("p_up", 0.33))
    p_stable   = float(row.get("p_stable", max(0.0, 1.0 - p_down - p_up)))
    mom_sig    = row.get("momentum_signal", "neutral")

    # Anomaly
    anomaly_score = int(row.get("anomaly_score", 0))
    z_anom     = row.get("z_anomaly", False)
    z_anom_60  = row.get("z_anomaly_60", False)
    if_anom    = row.get("if_anomaly", False)
    ae_anom    = row.get("ae_anomaly", False)
    z_score    = row.get("z_score", 0)
    ae_error   = row.get("ae_error", 0)
    mkt_wide   = row.get("is_market_wide", False)
    sec_wide   = row.get("is_sector_wide", False)

    # SHAP / explainability
    driver      = row.get("driver", "unknown")
    top3        = row.get("top3_shap", "")
    narrative   = row.get("narrative", "")
    conflict    = row.get("conflict", "")
    caution     = row.get("caution_flag", "")

    # Derived
    vol_ann = vol * 100 * 16

    detectors = []
    if z_anom:    detectors.append("Z-Score (30D)")
    if z_anom_60: detectors.append("Z-Score (60D)")
    if if_anom:   detectors.append("Isolation Forest")
    if ae_anom:   detectors.append("LSTM Autoencoder")

    scope = ("market-wide" if mkt_wide
             else "sector-wide" if sec_wide
             else "stock-specific")

    anomaly_detected = "Yes" if anomaly_score >= 1 else "No"
    models_used      = ", ".join(detectors) if detectors else "None"
    news_sentiment   = row.get("news_sentiment_score", row.get("sentiment_note", "neutral"))
    existing_summary = row.get("narrative", row.get("summary", ""))

    current_price   = row.get("current_price", row.get("Close", 0))
    ema_20          = row.get("ema_20", current_price)
    price_3m_high   = row.get("price_3m_high", current_price)
    price_3m_low    = row.get("price_3m_low", current_price)
    monthly_summary = row.get("monthly_summary", "N/A")
    support         = round(current_price * 0.92, 2)   # ~8% below as rough support

    ema_diff_pct    = ((current_price / ema_20) - 1) * 100 if ema_20 else 0
    ema_position    = (f"{ema_diff_pct:+.1f}% above EMA20 ({ema_20:.2f})" if ema_diff_pct > 0.3
                       else f"{abs(ema_diff_pct):.1f}% below EMA20 ({ema_20:.2f})" if ema_diff_pct < -0.3
                       else f"on EMA20 ({ema_20:.2f}) — critical level")

    # Investment strategy levels
    daily_vol       = vol if vol > 0 else 0.015
    entry_low       = round(ema_20 * (1 - daily_vol), 2)      # 1 daily-vol below EMA20
    entry_high      = round(ema_20 * (1 + daily_vol * 0.5), 2) # half daily-vol above EMA20
    exit_target     = round(price_3m_high * 0.98, 2)           # just below 3M resistance
    exit_spike      = round(price_3m_high, 2)                  # at 3M high = resistance

    # Reduction % based on p_down + risk confidence + anomaly
    _red_base = max(0, (p_down - 0.45) * 100)                  # 0% at p_down≤0.45, scales up
    _red_base += 10 if anomaly_score >= 2 else 0
    _red_base += 10 if row.get("p_high", 0) >= 0.65 else 0
    reduction_pct   = min(int(_red_base), 75)

    return f"""You are a sharp equity analyst. Write direct, actionable analysis.
{lang}

RULES:
- Use ONLY the numbers from DATA. Do not invent prices or percentages.
- Be specific: name exact price levels from the data.
- Write like a trader, not a textbook.
- Translate ALL section headers and text to the language specified above.

Output EXACTLY this structure (7 sections, translate headers):

## Current Situation
- **Price: {current_price:.2f} ({ret_1d*100:+.2f}% today)** [🔴 if negative, 🟢 if positive]
- Volume: [format as e.g. 1.57M] — [above/below average comment based on vol_ratio={vol_ratio:.1f}x]
- RSI: {rsi:.1f} — [interpretation: <30=oversold, 30-45=weak, 45-55=neutral, 55-70=strong, >70=overbought]
- Price is {ema_position} — [critical/normal comment]

---

## Chart Analysis
Using the monthly price history, describe the price cycle in 3-4 short lines:
```
[Month]: [e.g. Peak ~X.XX]
[Month]: [e.g. Low ~X.XX]
[Month]: [current situation]
```
One sentence: which phase are we currently in.

---

## Probabilities Next 1-2 Weeks

| Scenario | Probability | Condition |
|----------|-------------|-----------|
| 📉 **Further down** | **{p_down*100:.0f}%** | [specific: e.g. price breaks below EMA {ema_20:.2f}] |
| ➡️ **Sideways** | **{p_stable*100:.0f}%** | [range: e.g. holds {price_3m_low:.2f}–{price_3m_high:.2f}] |
| 📈 **Recovery** | **{p_up*100:.0f}%** | [specific: e.g. volume spike + close above X.XX] |

---

## Signals

**🔴 Bearish signals:**
[2-3 bullets from data: 1-day drop, momentum, news, market context]

**🟢 Positive factors:**
[2-3 bullets from data: RSI not oversold, support holding, regime, etc.]

---

## Decision Tree Tomorrow

```
Opens above {current_price * 1.02:.2f}: → [action]
Opens below {current_price * 0.97:.2f}: → [action, next support ~{support}]
Opens between: → wait
```

---

## Stop Loss
[One sentence: suggested stop loss level with brief rationale based on 3M low={price_3m_low:.2f} and volatility]

---

## Investment Strategy

**→ Not holding:**
Based on direction={direction.upper()} and support at EMA20={ema_20:.2f}:
- If looking to enter: wait for price to reach the zone **{entry_low:.2f} – {entry_high:.2f}** (EMA20 support zone)
- Only enter if volume confirms (above average) and RSI is not falling further
- Do NOT chase if price is already {ema_diff_pct:+.1f}% away from EMA20

**→ Holding:**
{"Reduce position by ~" + str(reduction_pct) + "% — P(down)=" + f"{p_down*100:.0f}%" + " with risk=" + row.get("severity","?") + ". Keep core position only." if reduction_pct >= 20
 else "Hold — no strong exit signal yet. Watch for price breaking below " + f"{entry_low:.2f}" + "."}

**→ Take profit / Sell signal:**
- Consider selling {f"20–30% at {exit_target:.2f} (near 3M resistance)" if p_down < 0.50 else f"50%+ at {exit_target:.2f} — high downside probability"}
- Full exit if price reaches {exit_spike:.2f} (3M high) AND P(down) > 50%
- [Add one sentence based on news sentiment and anomaly scope]

> ⚠️ *Probabilities based on technical analysis — not guaranteed.*

---

DATA:
Ticker: {row['ticker']}
Date: {row.get('date', 'latest')}
Current Price: {current_price:.4f}
EMA20: {ema_20:.4f} ({ema_position})
3M High: {price_3m_high:.4f} | 3M Low: {price_3m_low:.4f}
Monthly Price History: {monthly_summary}
Price Change (1D): {ret_1d*100:+.2f}%
RSI: {rsi:.1f}
Momentum (5D / 10D): {mom5:+.3f} / {mom10:+.3f}
Trend / Regime: {regime}
Volume vs 20D avg: {vol_ratio:.1f}x ({confirm})
OBV Signal: {obv:+.3f}
Annualised Volatility: {vol_ann:.1f}%
Max Drawdown 30D: {drawdown*100:.1f}%
Excess Return vs Market: {excess*100:+.2f}%
Anomaly Detected: {anomaly_detected} ({anomaly_score}/4 detectors triggered)
Models Used: {models_used}
Anomaly Scope: {scope}
AI Direction Forecast (5D): {direction.upper()} | P(down)={p_down*100:.0f}% | P(up)={p_up*100:.0f}%
Main Risk Driver: {driver}
Top Contributing Factors: {top3}
Risk Level: {row['severity']}
Recommended Action: {row['action']}
Signal Conflicts: {conflict if conflict else 'None'}
Caution Flags: {caution if caution else 'None'}
News Sentiment: {news_sentiment}"""


def summarize(row: dict, language: str = "english", retries: int = 4) -> str:
    """Generate full analysis for one ticker. Falls back to narrative_text on failure."""
    global _tokens_used

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return row.get("narrative_text", "")

    # Stop if daily budget is nearly exhausted
    if _tokens_used >= DAILY_TOKEN_BUDGET:
        logging.warning(f"[llm_narrator] Daily token budget ({DAILY_TOKEN_BUDGET:,}) reached — skipping {row.get('ticker', '?')}")
        return row.get("narrative_text", "")

    client = Groq(api_key=api_key)
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": _build_prompt(row, language)}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            used = getattr(response.usage, "total_tokens", MAX_TOKENS + 700)
            _tokens_used += used
            return response.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e)
            # Token-per-day limit hit — stop retrying entirely
            if "tokens per day" in err_str or "TPD" in err_str:
                logging.warning(f"[llm_narrator] Daily token limit hit — stopping narrator for today.")
                _tokens_used = DAILY_TOKEN_BUDGET  # mark budget as exhausted
                return row.get("narrative_text", "")
            wait = 2 ** attempt
            logging.warning(
                f"[llm_narrator] {row.get('ticker', '?')} attempt {attempt+1} failed: {e} "
                f"— retrying in {wait}s"
            )
            time.sleep(wait)

    logging.error(f"[llm_narrator] All retries failed for {row.get('ticker', '?')}, using fallback.")
    return row.get("narrative_text", "")


def run(
    explanations_path: str,
    language: str = "english",
    severity_filter=DEFAULT_SEVERITY_FILTER,
) -> pd.DataFrame:
    """
    Run LLM Narrator on explanations.parquet.
    Enriches each row with detection + decision data before calling the LLM.

    Args:
        explanations_path: path to explanations.parquet
        language:          "english" | "german" | "arabic"
        severity_filter:   only process these severities (None = all tickers)

    Returns:
        DataFrame with ticker + llm_summary columns.
    """
    df           = pd.read_parquet(explanations_path)
    detection_dir = ROOT / "data/detection"
    decisions_df  = None
    decisions_path = ROOT / "data/decisions/decisions.parquet"
    if decisions_path.exists():
        decisions_df = pd.read_parquet(decisions_path)

    if severity_filter:
        to_process = df[df["severity"].isin(severity_filter)]
        skipped    = df[~df["severity"].isin(severity_filter)].copy()
        skipped["llm_summary"] = skipped["narrative_text"]
    else:
        to_process = df
        skipped    = pd.DataFrame()

    print(f"\nLLM Narrator — {language.upper()}  |  model: {MODEL}")
    print(f"Processing {len(to_process)} tickers (filter: {severity_filter or 'all'})")
    print("=" * 65)

    cache   = _load_cache()
    results = []
    for _, row in to_process.iterrows():
        ticker   = row["ticker"]
        key      = _cache_key(ticker)

        # Cache hit — skip Groq entirely
        if key in cache:
            print(f"\n{ticker} [{row['severity']}] (cached)")
            results.append({"ticker": ticker, "llm_summary": cache[key]})
            continue

        row_dict = row.to_dict()

        # Enrich with latest detection data
        det = _load_detection_latest(ticker, detection_dir)
        row_dict.update(det)

        # Enrich with full decision data (direction, p_down, momentum_signal, caution_flag)
        if decisions_df is not None:
            dec_rows = decisions_df[decisions_df["ticker"] == ticker]
            if not dec_rows.empty:
                for col in [
                    "direction", "p_up", "p_stable", "p_down", "p_high",
                    "momentum_signal", "caution_flag", "summary",
                    "vader_score", "finbert_score", "news_sentiment_score",
                ]:
                    if col in dec_rows.columns:
                        row_dict[col] = dec_rows.iloc[0][col]

        summary = summarize(row_dict, language=language)
        cache[key] = summary
        _save_cache(cache)

        results.append({"ticker": ticker, "llm_summary": summary})
        print(f"\n{ticker} [{row['severity']}]")
        print(f"  {summary[:120]}...")
        time.sleep(2)   # respect Groq rate limit

    result_df = pd.DataFrame(results)

    if not skipped.empty:
        skipped_df = skipped[["ticker", "llm_summary"]].reset_index(drop=True)
        result_df  = pd.concat([result_df, skipped_df], ignore_index=True)

    out_path = ROOT / "data/explanations/llm_summaries.parquet"
    result_df.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")

    return result_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="english",
                        choices=["english", "german", "arabic"])
    parser.add_argument("--all", action="store_true",
                        help="Process all tickers, not just CRITICAL/WARNING")
    args = parser.parse_args()

    run(
        explanations_path=str(ROOT / "data/explanations/explanations.parquet"),
        language=args.language,
        severity_filter=None if args.all else DEFAULT_SEVERITY_FILTER,
    )
