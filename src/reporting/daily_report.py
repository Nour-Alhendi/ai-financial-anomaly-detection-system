"""
FinWatch AI — Layer 8B: Daily Summary Report
=============================================
Prints + saves a management-level summary of one pipeline run.

Output: data/reports/daily_summary.txt
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]

SEVERITY_ORDER = ["CRITICAL", "WARNING", "WATCH", "REVIEW", "POSITIVE_MOMENTUM", "NORMAL"]


def run(decisions_path=None) -> str:
    decisions_path = decisions_path or str(ROOT / "data/decisions/decisions.parquet")
    df = pd.read_parquet(decisions_path)

    counts   = Counter(df["severity"].tolist())
    date_str = datetime.today().strftime("%Y-%m-%d")

    lines = [
        "=" * 55,
        f"  FinWatch AI — Daily Summary ({date_str})",
        "=" * 55,
        f"  Total tickers monitored: {len(df)}",
        "",
        "  Severity Breakdown:",
    ]

    for sev in SEVERITY_ORDER:
        n = counts.get(sev, 0)
        if n:
            lines.append(f"    {sev:<20} {n:>3} ticker(s)")

    # Escalated tickers
    escalated = df[df["action"] == "ESCALATE"]
    if not escalated.empty:
        lines += ["", "  ESCALATE:", *[f"    - {r['ticker']}: {r['severity']} ({r['context']})"
                                        for _, r in escalated.iterrows()]]

    # Monitor tickers
    monitored = df[df["action"] == "MONITOR"]
    if not monitored.empty:
        lines += ["", "  MONITOR:", *[f"    - {r['ticker']}: {r['severity']}"
                                       for _, r in monitored.iterrows()]]

    # Positive signals
    positive = df[df["severity"] == "POSITIVE_MOMENTUM"]
    if not positive.empty:
        lines += ["", "  POSITIVE MOMENTUM:", *[f"    + {r['ticker']}"
                                                  for _, r in positive.iterrows()]]

    lines += ["", "=" * 55]
    report = "\n".join(lines)
    print(report)

    out_path = ROOT / "data/reports/daily_summary.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\n[daily_report] Saved: {out_path}")

    return report


if __name__ == "__main__":
    run()
