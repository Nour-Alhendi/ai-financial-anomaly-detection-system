"""
FinWatch AI — Portfolio Page (minimal redesign v2)
"""

import streamlit as st
import pandas as pd

from data.loader import (
    COMPANY_NAMES, SEV_COLOR, load_decisions, load_price_summary, load_detection,
)
from data.portfolio import (
    load_portfolios, save_portfolios,
    add_position, remove_position,
    create_portfolio, delete_portfolio,
)

ALL_TICKERS = list(COMPANY_NAMES.keys())

SEV_ICON = {
    "CRITICAL": "🔴", "WARNING": "🟡", "WATCH": "🔵",
    "NORMAL": "🟢", "POSITIVE_MOMENTUM": "🟢", "REVIEW": "🟣",
}

CSS = """
<style>
/* ── KPI cards ── */
.pf-kpi-label {
    font-size:8px;letter-spacing:2.5px;text-transform:uppercase;
    color:#3d5266;font-family:'IBM Plex Mono',monospace;margin-bottom:4px;
}
.pf-kpi-value {
    font-size:22px;font-weight:600;font-family:'IBM Plex Mono',monospace;color:#e2e8f0;
}
.pf-kpi-sub { font-size:10px;font-family:'IBM Plex Mono',monospace;color:#637a91;margin-top:2px; }
.pf-up   { color:#1de9b6!important; }
.pf-down { color:#f85149!important; }

/* ── Stock list rows ── */
.pos-stat-label {
    font-size:8px;letter-spacing:2px;text-transform:uppercase;
    color:#3d5266;font-family:'IBM Plex Mono',monospace;
}
.pos-stat-val {
    font-size:12px;font-weight:500;font-family:'IBM Plex Mono',monospace;color:#c9d1d9;
}

/* ── Section labels ── */
.section-lbl {
    font-size:8px;letter-spacing:2.5px;text-transform:uppercase;
    color:#3d5266;font-family:'IBM Plex Mono',monospace;
    border-bottom:1px solid rgba(30,45,65,0.4);padding-bottom:5px;margin-bottom:12px;
}

/* ── Strategy modal ── */
.s-section-head {
    font-size:7px;letter-spacing:2.5px;text-transform:uppercase;
    color:#3d5266;border-bottom:1px solid rgba(30,45,65,0.4);
    padding-bottom:4px;margin:12px 0 8px;
    font-family:'IBM Plex Mono',monospace;
}
.strategy-row {
    display:flex;gap:10px;padding:4px 0;font-size:10px;
    font-family:'IBM Plex Mono',monospace;line-height:1.7;
}
.s-arrow-up { color:#1de9b6;font-weight:700;white-space:nowrap;min-width:100px; }
.s-arrow-dn { color:#f85149;font-weight:700;white-space:nowrap;min-width:100px; }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_usd(val: float) -> str:
    if abs(val) >= 1_000_000: return f"${val/1_000_000:.2f}M"
    if abs(val) >= 1_000:     return f"${val/1_000:.1f}K"
    return f"${val:.2f}"


def _portfolio_series(positions: list) -> pd.DataFrame:
    """Build a daily portfolio-value time series from detection parquet files.

    Each position contributes shares × Close from its entry_date onward.
    Returns a DataFrame with columns [date, value, pct_change].
    """
    if not positions:
        return pd.DataFrame(columns=["date", "value"])

    combined: pd.Series | None = None

    for pos in positions:
        ticker     = pos["ticker"]
        entry_date = pd.Timestamp(pos["entry_date"])
        shares     = float(pos["shares"])
        try:
            det_df = load_detection(ticker)
            mask   = det_df["Date"] >= entry_date
            sub    = det_df.loc[mask, ["Date", "Close"]].copy()
            if sub.empty:
                continue
            s = (sub.set_index("Date")["Close"] * shares).rename(ticker)
            combined = s if combined is None else combined.add(s, fill_value=0)
        except Exception:
            continue

    if combined is None:
        return pd.DataFrame(columns=["date", "value"])

    df = combined.reset_index()
    df.columns = ["date", "value"]
    df = df.sort_values("date").reset_index(drop=True)

    # Add % change from start
    start = df["value"].iloc[0]
    df["pct_change"] = ((df["value"] - start) / start * 100) if start else 0.0
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def _portfolio_line_chart(df: pd.DataFrame) -> None:
    """Render an interactive portfolio value line chart via Vega-Lite."""
    if df.empty:
        return

    total_now   = df["value"].iloc[-1]
    total_start = df["value"].iloc[0]
    gain        = total_now - total_start
    gain_pct    = (gain / total_start * 100) if total_start else 0
    color       = "#1de9b6" if gain >= 0 else "#f85149"
    sign        = "+" if gain >= 0 else ""

    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;margin-bottom:8px">'
        f'<span style="font-size:24px;font-weight:700;color:#e2e8f0">{_fmt_usd(total_now)}</span>'
        f'&nbsp;&nbsp;<span style="font-size:13px;color:{color}">'
        f'{sign}{_fmt_usd(gain)}&nbsp;({sign}{gain_pct:.2f}%)</span>'
        f'<span style="font-size:9px;color:#3d5266;margin-left:10px">since first purchase</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "background": "#060a0f",
        "width": "container",
        "height": 240,
        "data": {"values": df.to_dict("records")},
        "layer": [
            {
                "mark": {
                    "type": "area",
                    "line": {"color": color, "strokeWidth": 2},
                    "color": {
                        "x1": 1, "y1": 1, "x2": 1, "y2": 0,
                        "gradient": "linear",
                        "stops": [
                            {"offset": 0, "color": "rgba(6,10,15,0)"},
                            {"offset": 1, "color": color},
                        ],
                    },
                    "opacity": 0.18,
                    "interpolate": "monotone",
                },
                "encoding": {
                    "x": {
                        "field": "date", "type": "temporal",
                        "axis": {
                            "labelColor": "#3d5266", "tickColor": "transparent",
                            "domainColor": "rgba(30,45,65,0.3)",
                            "labelFont": "IBM Plex Mono, monospace",
                            "labelFontSize": 9, "grid": False,
                            "format": "%b %d",
                        },
                        "title": None,
                    },
                    "y": {
                        "field": "value", "type": "quantitative",
                        "axis": {
                            "labelColor": "#3d5266", "tickColor": "transparent",
                            "domainColor": "transparent", "gridColor": "rgba(30,45,65,0.2)",
                            "labelFont": "IBM Plex Mono, monospace",
                            "labelFontSize": 9,
                            "format": "$,.0f",
                        },
                        "title": None,
                    },
                    "tooltip": [
                        {"field": "date",       "type": "temporal",    "title": "Date",     "format": "%Y-%m-%d"},
                        {"field": "value",      "type": "quantitative","title": "Value ($)","format": ",.2f"},
                        {"field": "pct_change", "type": "quantitative","title": "Change (%)", "format": "+.2f"},
                    ],
                },
            },
            # Vertical rule + point on hover
            {
                "transform": [{"filter": {"param": "hover", "empty": False}}],
                "mark": {"type": "rule", "color": "rgba(255,255,255,0.1)", "strokeWidth": 1},
                "encoding": {"x": {"field": "date", "type": "temporal"}},
            },
        ],
        "params": [{
            "name": "hover",
            "select": {"type": "point", "fields": ["date"], "nearest": True, "on": "mouseover"},
        }],
        "view": {"stroke": None},
        "config": {"background": "#060a0f"},
    }

    st.vega_lite_chart(df, spec, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Stock detail modal  (defined inside render_portfolio_page to avoid module-
# level @st.dialog, which crashes on import before Streamlit is running)
# ─────────────────────────────────────────────────────────────────────────────

def _render_stock_modal_ui(pos: dict, dec: dict, current_price: float, chg: float) -> None:
    """All modal UI — called from the @st.dialog wrapper inside render_portfolio_page."""
    ticker  = pos["ticker"]
    name    = COMPANY_NAMES.get(ticker, ticker)
    sev     = dec.get("severity", "NORMAL")
    sev_col = SEV_COLOR.get(sev, "#637a91")
    ts      = str(dec.get("trading_signal", "NEUTRAL"))
    ts_cfg  = {
        "ENTRY":   ("#1de9b6", "▲ ENTRY"),
        "EXIT":    ("#f85149", "▼ EXIT"),
        "HOLD":    ("#58a6ff", "◆ HOLD"),
        "NEUTRAL": ("#637a91", "— NEUTRAL"),
    }
    ts_col, ts_lbl = ts_cfg.get(ts, ts_cfg["NEUTRAL"])
    chg_col  = "#1de9b6" if chg >= 0 else "#f85149"
    chg_sign = "+" if chg >= 0 else ""
    pnl      = (current_price - pos["entry_price"]) * pos["shares"]
    pnl_pct  = ((current_price - pos["entry_price"]) / pos["entry_price"] * 100) if pos["entry_price"] else 0

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="margin-bottom:16px">'
        f'<div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;'
        f'color:#3d5266;font-family:IBM Plex Mono,monospace;margin-bottom:4px">'
        f'{name}</div>'
        f'<div style="display:flex;align-items:baseline;gap:12px">'
        f'<span style="font-size:28px;font-weight:700;color:#e2e8f0;font-family:IBM Plex Mono,monospace">'
        f'${current_price:.2f}</span>'
        f'<span style="font-size:13px;color:{chg_col};font-family:IBM Plex Mono,monospace">'
        f'{chg_sign}{chg:.2f}% today</span>'
        f'<span style="font-size:11px;font-weight:600;color:{sev_col};font-family:IBM Plex Mono,monospace">'
        f'● {sev}</span>'
        f'<span style="font-size:11px;font-weight:600;color:{ts_col};font-family:IBM Plex Mono,monospace">'
        f'{ts_lbl}</span>'
        f'</div>'
        f'<div style="font-size:10px;color:{"#1de9b6" if pnl>=0 else "#f85149"};'
        f'font-family:IBM Plex Mono,monospace;margin-top:4px">'
        f'{"+" if pnl>=0 else ""}{_fmt_usd(pnl)} ({pnl_pct:+.1f}%) · '
        f'{pos["shares"]:.0f} shares @ ${pos["entry_price"]:.2f} entry</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Price chart (90-day) ──────────────────────────────────────────────────
    try:
        det_df = load_detection(ticker)
        if det_df is not None and not det_df.empty:
            df90 = det_df.tail(90).copy()
            df90["date"]  = df90["Date"].dt.strftime("%Y-%m-%d")
            df90["close"] = df90["Close"]

            c_line  = "#1de9b6" if chg >= 0 else "#f85149"
            chart_spec = {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "background": "#060a0f",
                "width": "container", "height": 160,
                "data": {"values": df90[["date","close"]].to_dict("records")},
                "mark": {
                    "type": "area",
                    "line": {"color": c_line, "strokeWidth": 1.5},
                    "color": {
                        "x1": 1, "y1": 1, "x2": 1, "y2": 0,
                        "gradient": "linear",
                        "stops": [
                            {"offset": 0, "color": "rgba(6,10,15,0)"},
                            {"offset": 1, "color": c_line},
                        ],
                    },
                    "opacity": 0.15, "interpolate": "monotone",
                },
                "encoding": {
                    "x": {
                        "field": "date", "type": "temporal", "title": None,
                        "axis": {
                            "labelColor": "#3d5266", "tickColor": "transparent",
                            "domainColor": "rgba(30,45,65,0.3)",
                            "labelFont": "IBM Plex Mono, monospace",
                            "labelFontSize": 8, "grid": False, "format": "%b %d",
                        },
                    },
                    "y": {
                        "field": "close", "type": "quantitative", "title": None,
                        "axis": {
                            "labelColor": "#3d5266", "tickColor": "transparent",
                            "domainColor": "transparent",
                            "gridColor": "rgba(30,45,65,0.2)",
                            "labelFont": "IBM Plex Mono, monospace",
                            "labelFontSize": 8, "format": "$,.0f",
                        },
                    },
                    "tooltip": [
                        {"field": "date",  "type": "temporal",    "title": "Date",     "format": "%Y-%m-%d"},
                        {"field": "close", "type": "quantitative","title": "Close ($)","format": ",.2f"},
                    ],
                },
                "view": {"stroke": None},
                "config": {"background": "#060a0f"},
            }
            st.vega_lite_chart(df90[["date","close"]], chart_spec, use_container_width=True)
    except Exception:
        pass

    # ── AI Strategy ───────────────────────────────────────────────────────────
    strategy_html = _build_strategy(pos, dec, current_price)
    st.markdown(
        f'<div style="background:rgba(6,10,15,0.9);border:1px solid rgba(30,45,65,0.5);'
        f'border-radius:8px;padding:14px 18px;margin-top:10px;'
        f'font-family:IBM Plex Mono,monospace;font-size:10px;line-height:1.8;color:#8b9aab">'
        f'{strategy_html}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# AI Strategy builder (unchanged logic)
# ─────────────────────────────────────────────────────────────────────────────

def _build_strategy(pos: dict, decision: dict, current_price: float) -> str:  # noqa: C901
    shares        = pos["shares"]
    entry         = pos["entry_price"]
    ticker        = pos["ticker"]
    pnl_per       = current_price - entry
    pnl_total     = pnl_per * shares
    pnl_pct       = (pnl_per / entry) * 100 if entry else 0
    sev           = decision.get("severity", "NORMAL")
    p_high        = float(decision.get("p_high")  or 0.3)
    confidence    = float(decision.get("confidence") or 0.0)
    vader_score   = float(decision.get("vader_score")          or 0.0)
    finbert_score = float(decision.get("finbert_score")        or 0.0)
    news_sentiment= float(decision.get("news_sentiment_score") or 0.0)

    p_drawdown_cv = float(decision.get("p_drawdown") or 0.33)
    vol_est   = current_price * 0.015
    add_low   = round(current_price - vol_est, 2)
    add_high  = round(current_price + vol_est * 0.5, 2)
    _stop_raw = round(entry * 0.92, 2) if pnl_pct < 10 else round(entry * 1.02, 2)
    stop_breached = _stop_raw >= current_price
    stop      = round(current_price * 0.97, 2) if stop_breached else _stop_raw
    max_loss  = abs((current_price - stop) * shares)
    exit_target = round(current_price * 1.07, 2) if pnl_pct < 15 else round(entry * (1 + pnl_pct / 100 + 0.08), 2)
    exit_pct  = (exit_target / current_price - 1) * 100

    reduce_pct    = 0
    if p_drawdown_cv > 0.45:
        reduce_pct = min(int((p_drawdown_cv - 0.45) * 100) + (10 if p_high >= 0.65 else 0), 75)
    reduce_shares = int(shares * reduce_pct / 100)
    reduce_value  = reduce_shares * current_price

    rsi_cv      = float(decision.get("rsi")         or 50.0)
    mom5_raw    = float(decision.get("momentum_5")  or 0.0)
    mom10_raw   = float(decision.get("momentum_10") or 0.0)
    mom5_cv     = mom5_raw  if abs(mom5_raw)  <= 1.0 else mom5_raw  / 100
    mom10_cv    = mom10_raw if abs(mom10_raw) <= 1.0 else mom10_raw / 100
    obv_cv      = float(decision.get("obv_signal")   or 0.0)
    drawdown_cv = float(decision.get("drawdown")     or 0.0)
    anom_cv     = int(decision.get("anomaly_score")  or 0)
    mkt_cv      = bool(decision.get("market_anomaly") or False)
    sec_cv      = bool(decision.get("sector_anomaly") or False)
    excess_cv   = float(decision.get("excess_return") or 0.0)

    lines = []

    # ── AI CONVICTION ─────────────────────────────────────────────────────────
    lines.append('<div class="s-section-head">AI CONVICTION</div>')

    if sev == "CRITICAL":
        conv_pct   = min(int(p_drawdown_cv * 100) + 10, 95)
        conv_color = "#f85149"
        conv_label = "EXIT — Critical risk detected"
        conv_icon  = "↓"
    elif sev == "WARNING":
        conv_pct   = min(int(p_drawdown_cv * 100), 85)
        conv_color = "#e3b341"
        conv_label = "CAUTION — Elevated downside risk"
        conv_icon  = "→"
    elif sev == "POSITIVE_MOMENTUM":
        conv_pct   = 65
        conv_color = "#1de9b6"
        conv_label = "HOLD / ADD — Positive momentum"
        conv_icon  = "↑"
    elif sev == "WATCH":
        conv_pct   = max(int(p_drawdown_cv * 100), 40)
        conv_color = "#58a6ff"
        conv_label = "WATCH — Slightly elevated risk"
        conv_icon  = "→"
    else:
        conv_pct   = 50
        conv_color = "#637a91"
        conv_label = "HOLD — No strong signal"
        conv_icon  = "→"

    reasons = []
    if p_drawdown_cv >= 0.60:
        reasons.append(f"<strong>High drawdown risk:</strong> {p_drawdown_cv*100:.0f}% probability of a >5% drop in the next 10 days.")
    elif p_drawdown_cv >= 0.45:
        reasons.append(f"Elevated drawdown probability: {p_drawdown_cv*100:.0f}% — monitor closely.")
    else:
        reasons.append(f"Drawdown probability: {p_drawdown_cv*100:.0f}% — within normal range.")

    if anom_cv >= 3:
        scope = "market-wide" if mkt_cv else ("sector-wide" if sec_cv else "stock-specific")
        reasons.append(f"<strong>{anom_cv}/4 anomaly detectors</strong> flagged unusual behavior ({scope}).")
    elif anom_cv == 2:
        scope = "market-wide" if mkt_cv else ("sector-wide" if sec_cv else "stock-specific")
        reasons.append(f"2/4 anomaly models flagged an irregular pattern ({scope}).")
    elif anom_cv == 1:
        reasons.append("1/4 anomaly models flagged a weak signal.")

    if rsi_cv <= 30:
        reasons.append(f"RSI={rsi_cv:.0f} — oversold. Often followed by a bounce.")
    elif rsi_cv >= 70:
        reasons.append(f"RSI={rsi_cv:.0f} — overbought. Upward pressure may fade.")
    elif 40 <= rsi_cv <= 60:
        reasons.append(f"RSI={rsi_cv:.0f} — neutral, no extreme momentum signal.")

    if abs(mom5_cv) > 0.005:
        if mom5_cv < -0.02 and mom10_cv > 0.01:
            reasons.append(f"Short-term pullback ({mom5_cv*100:.1f}%) within positive medium-term trend (+{mom10_cv*100:.1f}%) — classic pullback.")
        elif mom5_cv > 0.02 and mom10_cv < -0.01:
            reasons.append(f"Short-term bounce (+{mom5_cv*100:.1f}%) against negative medium-term trend ({mom10_cv*100:.1f}%) — caution.")
        elif mom5_cv < -0.02 and mom10_cv < -0.01:
            reasons.append(f"Momentum negative short-term ({mom5_cv*100:.1f}%) and medium-term ({mom10_cv*100:.1f}%) — confirmed downtrend.")
        elif mom5_cv > 0.02 and mom10_cv > 0.01:
            reasons.append(f"Momentum positive short-term (+{mom5_cv*100:.1f}%) and medium-term (+{mom10_cv*100:.1f}%) — confirmed uptrend.")

    if drawdown_cv < -0.10:
        reasons.append(f"<strong>30D drawdown: {drawdown_cv*100:.1f}%</strong> — significant loss from recent peak.")
    elif drawdown_cv < -0.05:
        reasons.append(f"30D drawdown: {drawdown_cv*100:.1f}% — moderate correction.")

    if obv_cv < -0.5:
        reasons.append("Selling pressure confirmed in volume — distribution phase.")
    elif obv_cv > 0.5:
        reasons.append("Buying pressure confirmed in volume — accumulation signal.")

    lines.append(f"""
<div style="background:rgba(3,6,10,0.7);border-left:3px solid {conv_color};border-radius:0 8px 8px 0;padding:12px 16px;margin-bottom:8px">
  <div style="font-size:13px;font-weight:700;color:{conv_color};margin-bottom:8px">
    {conv_icon} <span style="font-size:16px">{conv_pct}%</span> conviction &mdash; {conv_label}
  </div>
  <div style="font-size:10px;color:#8b9aab;line-height:2">
    <strong style="color:#637a91;letter-spacing:1px;font-size:8px">WHY?</strong><br>
    {"<br>".join(f"· {r}" for r in reasons)}
  </div>
</div>""")

    # ── RECOMMENDATION ────────────────────────────────────────────────────────
    lines.append('<div class="s-section-head">RECOMMENDATION</div>')

    if sev == "CRITICAL" and p_drawdown_cv >= 0.55:
        rec_color = "#f85149"
        rec_label = "↓  REDUCE POSITION"
        rec_body  = (f"Sell {reduce_pct}% of your position — {reduce_shares} shares "
                     f"@ {_fmt_usd(current_price)} = <strong style='color:#f85149'>{_fmt_usd(reduce_value)}</strong> locked in. "
                     f"Critical risk with P(drawdown)={p_drawdown_cv*100:.0f}%.")
    elif sev == "WARNING" and p_drawdown_cv >= 0.45:
        rec_color = "#e3b341"
        rec_label = "→  MONITOR — CONSIDER REDUCING"
        rec_body  = (f"Consider selling {reduce_pct}% ({reduce_shares} shares ≈ {_fmt_usd(reduce_value)}). "
                     f"WARNING level, elevated downside probability ({p_drawdown_cv*100:.0f}%). "
                     f"Watch {_fmt_usd(entry)} as break-even defense.")
    elif sev == "POSITIVE_MOMENTUM":
        tp_shares = max(1, int(shares * 0.25))
        tp_val    = tp_shares * exit_target
        rec_color = "#1de9b6"
        rec_label = "↑  TAKE PARTIAL PROFIT"
        rec_body  = (f"Strong momentum. Consider selling {tp_shares} shares near {_fmt_usd(exit_target)} ≈ {_fmt_usd(tp_val)}. "
                     f"Hold remaining {int(shares - tp_shares)} shares for further upside.")
    elif pnl_pct >= 30 and sev not in ("CRITICAL", "WARNING"):
        tp_shares = max(1, int(shares * 0.3))
        rec_color = "#1de9b6"
        rec_label = "↑  SECURE GAINS"
        rec_body  = (f"Position up {pnl_pct:.1f}% (+{_fmt_usd(pnl_total)}). "
                     f"Sell {tp_shares} shares near {_fmt_usd(exit_target)} to lock in profits.")
    else:
        pnl_sign  = "+" if pnl_total >= 0 else ""
        rec_color = "#637a91"
        rec_label = "→  HOLD"
        rec_body  = (f"No strong exit signal. P&L: {pnl_sign}{_fmt_usd(pnl_total)} ({pnl_pct:+.1f}%). "
                     f"Drawdown prob: {p_drawdown_cv*100:.0f}%. Watch {_fmt_usd(entry)} as key support.")

    lines.append(
        f'<div class="strategy-row">'
        f'<span style="color:{rec_color};font-weight:700;white-space:nowrap;margin-right:8px">{rec_label}</span>'
        f'<span style="color:#c9d1d9">{rec_body}</span></div>'
    )

    # ── KEY PRICE LEVELS ──────────────────────────────────────────────────────
    lines.append('<div class="s-section-head">KEY PRICE LEVELS</div>')

    if sev in ("NORMAL", "POSITIVE_MOMENTUM") and p_drawdown_cv < 0.35:
        lines.append(
            f'<div class="strategy-row"><span class="s-arrow-up" style="white-space:nowrap">↑ ADD ZONE</span>'
            f'<span>{_fmt_usd(add_low)} – {_fmt_usd(add_high)} · wait for volume confirmation</span></div>'
        )
    lines.append(
        f'<div class="strategy-row"><span class="s-arrow-up" style="white-space:nowrap">✦ EXIT TARGET</span>'
        f'<span>{_fmt_usd(exit_target)} · +{exit_pct:.1f}% from current</span></div>'
    )
    if stop_breached:
        lines.append(
            f'<div class="strategy-row"><span class="s-arrow-dn" style="white-space:nowrap">⬡ STOP LOSS</span>'
            f'<span style="color:#f85149"><strong>Original stop already breached.</strong> '
            f'Emergency stop at {_fmt_usd(stop)} (–3% from current) · '
            f'max additional loss <strong>{_fmt_usd(max_loss)}</strong> on {int(shares)} shares</span></div>'
        )
    else:
        lines.append(
            f'<div class="strategy-row"><span class="s-arrow-dn" style="white-space:nowrap">⬡ STOP LOSS</span>'
            f'<span>{_fmt_usd(stop)} · max loss <strong style="color:#f85149">{_fmt_usd(max_loss)}</strong> on {int(shares)} shares</span></div>'
        )

    # ── NEWS & MARKET INTELLIGENCE ────────────────────────────────────────────
    lines.append('<div class="s-section-head">NEWS &amp; MARKET INTELLIGENCE</div>')

    fetched_at    = decision.get("sentiment_fetched_at")
    staleness_html = ""
    if fetched_at:
        try:
            from datetime import datetime as _dt
            fetched_dt    = _dt.fromisoformat(str(fetched_at))
            news_age_days = (_dt.now() - fetched_dt).days
            if news_age_days >= 2:
                staleness_html = (
                    f'<div style="background:rgba(227,179,65,0.08);border:1px solid rgba(227,179,65,0.25);'
                    f'border-radius:5px;padding:6px 10px;margin-bottom:8px;'
                    f'font-size:9px;color:#e3b341;font-family:\'IBM Plex Mono\',monospace">'
                    f'⚠ News data is <strong>{news_age_days} days old</strong> — re-run the pipeline for fresh data.</div>'
                )
        except Exception:
            pass

    has_news = abs(news_sentiment) > 0.01 or abs(vader_score) > 0.01 or abs(finbert_score) > 0.01
    if has_news:
        lines.append(staleness_html)
        if news_sentiment >= 0.20:
            tone, tone_color = "strongly positive", "#1de9b6"
        elif news_sentiment >= 0.05:
            tone, tone_color = "moderately positive", "#58a6ff"
        elif news_sentiment <= -0.20:
            tone, tone_color = "strongly negative", "#f85149"
        elif news_sentiment <= -0.05:
            tone, tone_color = "moderately negative", "#e3b341"
        else:
            tone, tone_color = "neutral", "#637a91"

        pnl_exposure = abs(pnl_total)
        risk_ctx     = decision.get("context", "")

        rsi      = float(decision.get("rsi", 50) or 50)
        mom5     = mom5_cv
        mom10    = mom10_cv
        obv      = obv_cv
        drawdown = drawdown_cv
        excess   = excess_cv
        anom_sc  = anom_cv
        mkt_an   = mkt_cv
        sec_an   = sec_cv

        if rsi <= 30:
            rsi_text, rsi_color = f"RSI at {rsi:.0f} — oversold, recovery signal", "#1de9b6"
        elif rsi >= 70:
            rsi_text, rsi_color = f"RSI at {rsi:.0f} — overbought, correction risk", "#f85149"
        else:
            rsi_text, rsi_color = f"RSI at {rsi:.0f} — neutral momentum", "#637a91"

        if mom5 > 0.03 and mom10 > 0.03:
            mom_text, mom_color = f"Short and medium-term momentum both positive (+{mom5*100:.1f}% / +{mom10*100:.1f}%) — sustained uptrend", "#1de9b6"
        elif mom5 < -0.03 and mom10 < -0.03:
            mom_text, mom_color = f"Short and medium-term momentum both negative ({mom5*100:.1f}% / {mom10*100:.1f}%) — sustained downtrend", "#f85149"
        elif mom5 > 0.01 and mom10 < -0.01:
            mom_text, mom_color = f"Short-term bounce (+{mom5*100:.1f}%) against negative medium-term trend ({mom10*100:.1f}%) — caution", "#e3b341"
        elif mom5 < -0.01 and mom10 > 0.01:
            mom_text, mom_color = f"Short-term pullback ({mom5*100:.1f}%) within positive medium-term trend (+{mom10*100:.1f}%) — potential entry", "#58a6ff"
        else:
            mom_text, mom_color = f"Momentum flat (5d: {mom5*100:.1f}%, 10d: {mom10*100:.1f}%)", "#637a91"

        if obv > 0.5:    obv_text, obv_color = "Strong buying pressure — institutional accumulation likely", "#1de9b6"
        elif obv < -0.5: obv_text, obv_color = "Selling pressure confirmed — distribution phase, volume-backed decline", "#f85149"
        elif obv > 0.1:  obv_text, obv_color = "Mild buying pressure — volume supports price action", "#58a6ff"
        elif obv < -0.1: obv_text, obv_color = "Mild selling pressure — watch for volume acceleration", "#e3b341"
        else:            obv_text, obv_color = "Neutral volume pressure", "#637a91"

        dd_pct = drawdown * 100
        ex_pct = excess * 100
        if drawdown < -0.15:   dd_text, dd_color = f"Severe 30D drawdown of {dd_pct:.1f}% — significant peak-to-trough loss", "#f85149"
        elif drawdown < -0.08: dd_text, dd_color = f"Elevated drawdown of {dd_pct:.1f}% over 30 days", "#e3b341"
        else:                  dd_text, dd_color = f"Contained drawdown of {dd_pct:.1f}%", "#637a91"

        if ex_pct < -5:   vs_market, vm_color = f"Underperforming market by {abs(ex_pct):.1f}% — idiosyncratic weakness", "#f85149"
        elif ex_pct > 5:  vs_market, vm_color = f"Outperforming market by +{ex_pct:.1f}% — alpha generation", "#1de9b6"
        else:             vs_market, vm_color = f"Tracking market closely (excess return: {ex_pct:+.1f}%)", "#637a91"

        if anom_sc >= 3:   anom_text, anom_color = f"Strong anomaly signal — {anom_sc}/4 detection models flagged unusual activity", "#f85149"
        elif anom_sc >= 2: anom_text, anom_color = f"Moderate anomaly — {anom_sc}/4 models detected unusual behavior", "#e3b341"
        elif anom_sc == 1: anom_text, anom_color = "Weak anomaly signal — 1/4 model flagged, worth monitoring", "#58a6ff"
        else:              anom_text, anom_color = "No anomaly detected — price behavior within historical norms", "#637a91"

        scope_parts = []
        if mkt_an: scope_parts.append("market-wide")
        if sec_an: scope_parts.append("sector-wide")
        anom_scope = f" ({', '.join(scope_parts)} event)" if scope_parts else " (idiosyncratic)"
        if anom_sc > 0: anom_text += anom_scope

        lines.append(f"""
<div style="background:rgba(3,6,10,0.6);border-left:2px solid rgba(30,45,65,0.8);border-radius:0 6px 6px 0;padding:10px 14px;margin-top:6px;line-height:1.9;color:#8b9aab;font-size:10px">
  <div style="color:{tone_color};font-size:11px;font-weight:700;margin-bottom:6px">
    ↘ Recent news flow for {ticker} is <em>{tone}</em> — Combined Score: {news_sentiment:+.3f}
  </div>
  <div style="margin-bottom:4px">VADER {vader_score:+.2f} &nbsp;·&nbsp; Groq {finbert_score:+.2f} — models {"largely agree" if abs(vader_score - finbert_score) < 0.3 else "diverge"} on a {tone} news environment.</div>
  <div style="margin-bottom:4px">Current news sentiment {"moderately " if abs(news_sentiment) < 0.3 else "strongly "}weighs on {_fmt_usd(pnl_exposure)} of P&L exposure.</div>
  <div style="margin-bottom:5px"><span style="color:{rsi_color};font-weight:600">RSI &amp; Momentum — </span>{rsi_text}. {mom_text}.</div>
  <div style="margin-bottom:5px"><span style="color:{obv_color};font-weight:600">Volume — </span>{obv_text}.</div>
  <div style="margin-bottom:5px"><span style="color:{dd_color};font-weight:600">Drawdown — </span>{dd_text}. {vs_market}.</div>
  <div style="margin-bottom:5px"><span style="color:{anom_color};font-weight:600">Anomaly — </span>{anom_text}.</div>
  <div style="display:flex;gap:14px;margin-top:6px;font-size:9px;font-family:'IBM Plex Mono',monospace;color:#3d5266">
    <span>VADER <span style="color:#c9d1d9">{vader_score:+.3f}</span></span>
    <span>GROQ <span style="color:#c9d1d9">{finbert_score:+.3f}</span></span>
    <span>COMBINED <span style="color:#c9d1d9">{news_sentiment:+.3f}</span></span>
    <span>RISK CTX <span style="color:#c9d1d9">{risk_ctx}</span></span>
  </div>
</div>""")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main page
# ─────────────────────────────────────────────────────────────────────────────

def render_portfolio_page():
    st.markdown(CSS, unsafe_allow_html=True)

    if "portfolios"        not in st.session_state: st.session_state.portfolios        = load_portfolios()
    if "active_portfolio"  not in st.session_state: st.session_state.active_portfolio  = None
    if "show_add_form"     not in st.session_state: st.session_state.show_add_form     = False
    if "_pf_detail"        not in st.session_state: st.session_state._pf_detail        = None

    # ── Stock detail view (replaces dialog) ──────────────────────────────────
    if st.session_state._pf_detail is not None:
        d = st.session_state._pf_detail
        if st.button("← Back to Portfolio", key="btn_detail_back"):
            st.session_state._pf_detail = None
            st.rerun()
        _render_stock_modal_ui(d["pos"], d["dec"], d["price"], d["chg"])
        return

    portfolios   = st.session_state.portfolios
    decisions_df = load_decisions()
    price_data   = load_price_summary()
    dec_map      = {r["ticker"]: r.to_dict() for _, r in decisions_df.iterrows()}

    # ── Top bar: New Portfolio ────────────────────────────────────────────────
    top_l, top_r = st.columns([5, 1], gap="small")
    with top_r:
        if st.button("＋ New Portfolio", use_container_width=True, key="btn_new_pf"):
            st.session_state._creating_pf = True

    # ── New portfolio name form ───────────────────────────────────────────────
    if st.session_state.get("_creating_pf"):
        with top_l:
            new_name = st.text_input("", placeholder="Portfolio name, e.g. Tech Growth",
                                     key="new_pf_name", label_visibility="collapsed")
        c1, c2, _ = st.columns([1, 1, 4], gap="small")
        with c1:
            if st.button("Create", use_container_width=True, key="btn_create_confirm"):
                if new_name.strip():
                    portfolios = create_portfolio(portfolios, new_name.strip())
                    save_portfolios(portfolios)
                    st.session_state.portfolios  = portfolios
                    st.session_state._creating_pf = False
                    st.rerun()
        with c2:
            if st.button("Cancel", use_container_width=True, key="btn_create_cancel"):
                st.session_state._creating_pf = False
                st.rerun()

    names = list(portfolios.keys())
    if not names:
        st.markdown(
            '<div style="padding:48px 0;text-align:center;font-family:IBM Plex Mono;'
            'font-size:11px;color:#3d5266">No portfolios yet — create one above.</div>',
            unsafe_allow_html=True,
        )
        return

    sev_order = {"CRITICAL":0,"WARNING":1,"WATCH":2,"REVIEW":3,"POSITIVE_MOMENTUM":4,"NORMAL":5}

    # ── One expander card per portfolio ──────────────────────────────────────
    for pf_name in names:
        positions = portfolios.get(pf_name, [])

        # Enrich positions for summary line
        enriched    = []
        total_value = 0.0
        total_cost  = 0.0
        for pos in positions:
            ticker = pos["ticker"]
            price, chg = price_data.get(ticker, (pos["entry_price"], 0.0))
            dec    = dec_map.get(ticker, {})
            value  = price * pos["shares"]
            cost   = pos["entry_price"] * pos["shares"]
            pnl    = value - cost
            pnl_pct = (pnl / cost * 100) if cost else 0
            sev    = dec.get("severity", "NORMAL")
            total_value += value
            total_cost  += cost
            enriched.append({**pos, "price": price, "chg": chg, "value": value,
                              "pnl": pnl, "pnl_pct": pnl_pct, "sev": sev, "dec": dec})

        enriched.sort(key=lambda x: sev_order.get(x["sev"], 9))
        total_pnl     = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost else 0
        pnl_sign      = "+" if total_pnl >= 0 else ""
        pnl_col       = "#1de9b6" if total_pnl >= 0 else "#f85149"
        n_pos         = len(positions)
        n_risk        = sum(1 for e in enriched if e["sev"] in ("CRITICAL", "WARNING"))

        # Build expander label
        risk_badge = f"  🔴 {n_risk} at risk" if n_risk else ""
        exp_label  = (
            f"{pf_name}   ·   {_fmt_usd(total_value)}"
            f"   {pnl_sign}{_fmt_usd(total_pnl)} ({pnl_sign}{total_pnl_pct:.1f}%)"
            f"   ·   {n_pos} position{'s' if n_pos!=1 else ''}{risk_badge}"
            if positions else f"{pf_name}   ·   empty"
        )

        with st.expander(exp_label, expanded=False):

            # ── Chart ─────────────────────────────────────────────────────────
            if enriched:
                series_df = _portfolio_series(positions)
                if not series_df.empty:
                    _portfolio_line_chart(series_df)

            # ── Stock rows ────────────────────────────────────────────────────
            if enriched:
                st.markdown(
                    '<div style="border-top:1px solid rgba(30,45,65,0.35);margin:8px 0 4px"></div>',
                    unsafe_allow_html=True,
                )
                for e in enriched:
                    ticker   = e["ticker"]
                    name     = COMPANY_NAMES.get(ticker, ticker)
                    sev      = e["sev"]
                    dec      = e["dec"]
                    price    = e["price"]
                    chg      = e["chg"]
                    pnl_e    = e["pnl"]
                    pnl_pct_e= e["pnl_pct"]
                    value_e  = e["value"]
                    weight   = (value_e / total_value * 100) if total_value > 0 else 0
                    sev_col  = SEV_COLOR.get(sev, "#637a91")
                    pnl_col_e= "#1de9b6" if pnl_e >= 0 else "#f85149"
                    chg_col  = "#1de9b6" if chg >= 0 else "#f85149"
                    chg_sign = "+" if chg >= 0 else ""
                    ts       = str(dec.get("trading_signal", "NEUTRAL"))
                    ts_cfg   = {"ENTRY":("#1de9b6","▲"),"EXIT":("#f85149","▼"),"HOLD":("#58a6ff","◆"),"NEUTRAL":("#637a91","—")}
                    ts_col, ts_ico = ts_cfg.get(ts, ts_cfg["NEUTRAL"])

                    r1, r2, r3, r4, r5, r6, r_det, r_rm = st.columns(
                        [1.6, 2.8, 1.5, 1.6, 1.6, 1.3, 1.1, 0.5], gap="small"
                    )
                    with r1:
                        st.markdown(
                            f'<div style="padding:6px 0">'
                            f'<div style="font-size:8px;letter-spacing:1.5px;text-transform:uppercase;'
                            f'color:{sev_col};font-family:IBM Plex Mono,monospace">'
                            f'{SEV_ICON.get(sev,"⚪")} {sev}</div>'
                            f'<div style="font-size:14px;font-weight:700;color:#e2e8f0;'
                            f'font-family:IBM Plex Mono,monospace;margin-top:1px">{ticker}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    with r2:
                        st.markdown(
                            f'<div style="padding:6px 0">'
                            f'<div class="pos-stat-label">Company</div>'
                            f'<div style="font-size:11px;color:#8b9aab;font-family:IBM Plex Mono,monospace">'
                            f'{name}</div></div>',
                            unsafe_allow_html=True,
                        )
                    with r3:
                        st.markdown(
                            f'<div style="padding:6px 0">'
                            f'<div class="pos-stat-label">Price</div>'
                            f'<div class="pos-stat-val">${price:.2f}</div>'
                            f'<div style="font-size:10px;color:{chg_col};font-family:IBM Plex Mono,monospace">'
                            f'{chg_sign}{chg:.2f}%</div></div>',
                            unsafe_allow_html=True,
                        )
                    with r4:
                        st.markdown(
                            f'<div style="padding:6px 0">'
                            f'<div class="pos-stat-label">Value · Weight</div>'
                            f'<div class="pos-stat-val">{_fmt_usd(value_e)}</div>'
                            f'<div style="font-size:10px;color:#3d5266;font-family:IBM Plex Mono,monospace">'
                            f'{weight:.1f}%</div></div>',
                            unsafe_allow_html=True,
                        )
                    with r5:
                        st.markdown(
                            f'<div style="padding:6px 0">'
                            f'<div class="pos-stat-label">P&amp;L</div>'
                            f'<div class="pos-stat-val" style="color:{pnl_col_e}">'
                            f'{"+" if pnl_e>=0 else ""}{_fmt_usd(pnl_e)}</div>'
                            f'<div style="font-size:10px;color:{pnl_col_e};font-family:IBM Plex Mono,monospace">'
                            f'{pnl_pct_e:+.1f}%</div></div>',
                            unsafe_allow_html=True,
                        )
                    with r6:
                        st.markdown(
                            f'<div style="padding:6px 0">'
                            f'<div class="pos-stat-label">Signal</div>'
                            f'<div style="font-size:12px;font-weight:700;color:{ts_col};'
                            f'font-family:IBM Plex Mono,monospace">{ts_ico} {ts}</div></div>',
                            unsafe_allow_html=True,
                        )
                    with r_det:
                        st.markdown("<div style='padding-top:10px'>", unsafe_allow_html=True)
                        if st.button("Details →", key=f"det_{ticker}_{pf_name}", use_container_width=True):
                            st.session_state._pf_detail = {"pos": e, "dec": dec, "price": price, "chg": chg}
                            st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)
                    with r_rm:
                        st.markdown("<div style='padding-top:10px'>", unsafe_allow_html=True)
                        if st.button("✕", key=f"rm_{ticker}_{pf_name}", help="Remove"):
                            portfolios = remove_position(portfolios, pf_name, ticker)
                            save_portfolios(portfolios)
                            st.session_state.portfolios = portfolios
                            st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown(
                        '<div style="border-bottom:1px solid rgba(30,45,65,0.2);margin:2px 0"></div>',
                        unsafe_allow_html=True,
                    )

            # ── Footer: Add + Delete ──────────────────────────────────────────
            st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
            add_key  = f"add_toggle_{pf_name}"
            if add_key not in st.session_state: st.session_state[add_key] = False

            foot_l, foot_m, foot_r = st.columns([1.4, 1.2, 5], gap="small")
            with foot_l:
                if st.button("＋ Add Position", key=f"btn_add_toggle_{pf_name}", use_container_width=True):
                    st.session_state[add_key] = not st.session_state[add_key]
            with foot_m:
                if st.button("🗑 Delete Portfolio", key=f"btn_del_{pf_name}", use_container_width=True):
                    portfolios = delete_portfolio(portfolios, pf_name)
                    save_portfolios(portfolios)
                    st.session_state.portfolios = portfolios
                    st.rerun()

            if st.session_state[add_key]:
                st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
                f1, f2 = st.columns([3, 2], gap="small")
                with f1:
                    t_sel = st.selectbox("Ticker", ALL_TICKERS,
                                         format_func=lambda t: f"{t} — {COMPANY_NAMES.get(t,'')}",
                                         label_visibility="collapsed", key=f"form_ticker_{pf_name}")
                with f2:
                    n_shares = st.number_input("Shares", min_value=0.01, value=10.0, step=1.0,
                                               label_visibility="collapsed", key=f"form_shares_{pf_name}")
                f3, f4 = st.columns([2, 2], gap="small")
                with f3:
                    cur_px = price_data.get(t_sel, (0.0, 0.0))[0]
                    e_px   = st.number_input("Entry Price ($)", min_value=0.01,
                                             value=float(round(cur_px, 2)) if cur_px else 1.0,
                                             step=0.01, label_visibility="collapsed",
                                             key=f"form_entry_{pf_name}")
                with f4:
                    e_date = st.date_input("Entry Date", label_visibility="collapsed",
                                           key=f"form_date_{pf_name}")
                if st.button("＋ Confirm Add", key=f"btn_add_confirm_{pf_name}", use_container_width=True):
                    portfolios = add_position(portfolios, pf_name, t_sel, n_shares, e_px, str(e_date))
                    save_portfolios(portfolios)
                    st.session_state.portfolios = portfolios
                    st.session_state[add_key]   = False
                    st.rerun()
