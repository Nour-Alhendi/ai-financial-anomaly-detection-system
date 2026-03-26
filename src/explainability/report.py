"""
FinWatch AI — Layer 7D: HTML Report
===================================
Generates a daily report from explanations + LLM + FinBERT outputs.
Uses Jinja2 for safe HTML rendering (XSS-safe via autoescaping).
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from jinja2 import Environment, BaseLoader

ROOT = Path(__file__).resolve().parents[2]

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>FinWatch AI Daily Report</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 960px; margin: 40px auto; color: #222; }
    h1   { border-bottom: 2px solid #333; padding-bottom: 8px; }
    h2   { margin-top: 32px; color: #1a3a5c; }
    p    { margin: 4px 0; }
    ul   { margin: 6px 0 0 20px; }
    li   { margin: 2px 0; }
    .sentiment-positive { color: green; }
    .sentiment-negative { color: red; }
    .sentiment-neutral  { color: gray; }
    hr   { border: none; border-top: 1px solid #ddd; margin: 16px 0; }
  </style>
</head>
<body>
  <h1>FinWatch AI Daily Report — {{ date }}</h1>
  <p>Total tickers: {{ tickers | length }}</p>

  {% for row in tickers %}
  <hr>
  <h2>{{ row.ticker }}</h2>
  <p><b>LLM Summary:</b> {{ row.llm_summary }}</p>
  {% if row.top_news %}
  <ul>
    {% for news, sentiment in row.news_items %}
    <li>{{ news }} — <i class="sentiment-{{ sentiment }}">{{ sentiment }}</i></li>
    {% endfor %}
  </ul>
  {% endif %}
  {% endfor %}
</body>
</html>
"""


def run():
    news_path = ROOT / "data/explanations/llm_news_enriched.parquet"
    llm_path  = ROOT / "data/explanations/llm_summaries.parquet"

    if news_path.exists():
        df = pd.read_parquet(news_path)
    else:
        df = pd.read_parquet(llm_path)
        df["top_news"]       = [[] for _ in range(len(df))]
        df["news_sentiment"] = [[] for _ in range(len(df))]

    tickers = []
    for _, row in df.iterrows():
        top_news       = row["top_news"] if isinstance(row["top_news"], list) else []
        news_sentiment = row["news_sentiment"] if isinstance(row["news_sentiment"], list) else []
        tickers.append({
            "ticker":     row["ticker"],
            "llm_summary": row["llm_summary"],
            "top_news":   top_news,
            "news_items": list(zip(top_news, news_sentiment)),
        })

    env      = Environment(loader=BaseLoader(), autoescape=True)
    template = env.from_string(TEMPLATE)
    html     = template.render(date=datetime.today().strftime("%Y-%m-%d"), tickers=tickers)

    out_path = ROOT / "data/reports/daily_report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Report saved: {out_path}")


if __name__ == "__main__":
    run()
