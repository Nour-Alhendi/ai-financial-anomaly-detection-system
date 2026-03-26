"""
FinWatch AI — Layer 7C: LLM Narrator
======================================
Takes structured Narrative Engine output and produces a plain-language summary
using Groq (Llama 3) — free tier, no hallucination possible (fully structured input).

Supported languages: english, german, arabic

Performance: by default only processes CRITICAL + WARNING tickers to stay
within Groq free tier limits (30 req/min). Use severity_filter=None for all.
"""

import os
import time
import logging
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MODEL       = "llama-3.1-8b-instant"
TEMPERATURE = 0.3
MAX_TOKENS  = 100

LANGUAGE_INSTRUCTION = {
    "english": "Respond in English.",
    "german":  "Antworte auf Deutsch.",
    "arabic":  "أجب باللغة العربية.",
}

# Only these severities are sent to LLM by default (speed + cost control)
DEFAULT_SEVERITY_FILTER = {"CRITICAL", "WARNING"}


def _build_prompt(row: dict, language: str) -> str:
    lang = LANGUAGE_INSTRUCTION.get(language, LANGUAGE_INSTRUCTION["english"])

    # Anomaly context: how many models confirmed + scope
    confidence_pct = int(row.get("confidence", 0) * 100)
    anomaly_context = f"{confidence_pct}% of anomaly detection models confirmed"
    scope = row.get("decision_context", "idiosyncratic")

    # Conflict inline
    conflict_note = f" [CONFLICT: {row['conflict']}]" if row.get("conflict") else ""

    return f"""You are a financial risk analyst. Write ONE sentence for a manager.
{lang} No jargon. Use only the data below.

Ticker: {row['ticker']} | Severity: {row['severity']}{conflict_note}
Driver: {row['driver']} (SHAP {row['driver_shap']:+.3f}) | Top3: {row['top3_shap']}
OBV: {row['obv_signal']:.3f} ({row['confirmation']}) | Signal: {row['narrative']}
Anomaly: {anomaly_context}, scope: {scope}

Summary:"""


def summarize(row: dict, language: str = "english", retries: int = 4) -> str:
    """
    Generate a one-sentence plain-language summary for one ticker.
    Falls back to narrative_text if no API key is set or all retries fail.
    Uses exponential backoff on rate-limit errors.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
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
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt
            logging.warning(f"[llm_narrator] {row.get('ticker', '?')} attempt {attempt+1} failed: {e} — retrying in {wait}s")
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

    Args:
        explanations_path: path to explanations.parquet
        language:          "english" | "german" | "arabic"
        severity_filter:   only process these severities (None = all tickers)

    Returns:
        DataFrame with ticker + llm_summary columns (ready for Streamlit).
    """
    df = pd.read_parquet(explanations_path)

    if severity_filter:
        to_process = df[df["severity"].isin(severity_filter)]
        skipped    = df[~df["severity"].isin(severity_filter)].copy()
        skipped["llm_summary"] = skipped["narrative_text"]
    else:
        to_process = df
        skipped    = pd.DataFrame()

    print(f"\nLLM Narrator — {language.upper()}")
    print(f"Processing {len(to_process)} tickers (filter: {severity_filter or 'all'})")
    print("=" * 65)

    results = []
    for _, row in to_process.iterrows():
        summary = summarize(row.to_dict(), language=language)
        results.append({"ticker": row["ticker"], "llm_summary": summary})
        print(f"\n{row['ticker']} [{row['severity']}]")
        print(f"  {summary}")

    result_df = pd.DataFrame(results)

    # Merge skipped tickers back in
    if not skipped.empty:
        skipped_df = skipped[["ticker", "llm_summary"]].reset_index(drop=True)
        result_df  = pd.concat([result_df, skipped_df], ignore_index=True)

    # Save
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
