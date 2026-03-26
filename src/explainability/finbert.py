"""
FinWatch AI — Layer 7E: News + FinBERT
=======================================
Takes LLM Narrator output and enriches it with news sentiment using FinBERT.

Output: ticker + llm_summary + top_news + news_sentiment
"""

import os
import time
import logging
import pandas as pd
from pathlib import Path
import torch
import requests

ROOT = Path(__file__).resolve().parents[2]

FINBERT_LABELS  = ["neutral", "positive", "negative"]
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
NEWS_ENDPOINT   = "https://finnhub.io/api/v1/company-news"

# Lazy-loaded singletons — populated on first call to enrich_with_news()
_TOKENIZER = None
_MODEL     = None


def _load_model():
    """Load FinBERT once and cache in module-level singletons."""
    global _TOKENIZER, _MODEL
    if _TOKENIZER is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print("Loading FinBERT model (first use)...")
        _TOKENIZER = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        _MODEL     = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        print("FinBERT model ready.")


def fetch_news(ticker: str, limit: int = 3, retries: int = 3) -> list[str]:
    if not FINNHUB_API_KEY:
        return []

    params = {
        "symbol": ticker,
        "from":   "2026-03-01",
        "to":     "2026-03-25",
        "token":  FINNHUB_API_KEY,
    }

    for attempt in range(retries):
        try:
            r = requests.get(NEWS_ENDPOINT, params=params, timeout=10)
            r.raise_for_status()
            articles = r.json()
            if not isinstance(articles, list):
                logging.warning(f"[finbert] Unexpected response for {ticker}: {articles}")
                return []
            return [a["headline"] for a in articles[:limit] if "headline" in a]
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            if status == 429:
                wait = 2 ** attempt
                logging.warning(f"[finbert] Rate-limited for {ticker}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                logging.error(f"[finbert] HTTP {status} for {ticker}: {e}")
                return []
        except Exception as e:
            logging.error(f"[finbert] Request failed for {ticker}: {e}")
            return []

    logging.error(f"[finbert] All {retries} retries failed for {ticker}.")
    return []


def analyze_sentiment(headline: str) -> str:
    _load_model()
    inputs = _TOKENIZER(headline, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = _MODEL(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return FINBERT_LABELS[pred]


def enrich_with_news(df: pd.DataFrame) -> pd.DataFrame:
    """Adds top news + FinBERT sentiment to LLM output."""
    enriched = []
    for _, row in df.iterrows():
        ticker    = row["ticker"]
        news_list = fetch_news(ticker)
        sentiments = [analyze_sentiment(h) for h in news_list]
        enriched.append({
            "ticker":         ticker,
            "llm_summary":    row["llm_summary"],
            "top_news":       news_list,
            "news_sentiment": sentiments,
        })
    return pd.DataFrame(enriched)


if __name__ == "__main__":
    llm_path = ROOT / "data/explanations/llm_summaries.parquet"
    df_llm   = pd.read_parquet(llm_path)

    enriched_df = enrich_with_news(df_llm)
    out_path    = ROOT / "data/explanations/llm_news_enriched.parquet"
    enriched_df.to_parquet(out_path, index=False)
    print(f"Saved enriched data: {out_path}")
