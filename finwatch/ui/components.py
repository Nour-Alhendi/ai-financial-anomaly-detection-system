"""
FinWatch AI — UI Components
==============================
Sidebar, stock header, risk/news row, anomaly selector,
analysis panel, investor summary, and LLM report renderer.
"""

import ast
import re
import html as _html
import urllib.parse
import streamlit as st
import pandas as pd

from data.loader import (
    COMPANY_NAMES, SECTORS, SEV_CLS, load_anomaly_precision,
)
from analytics.analysis import (
    build_analysis, build_investor_summary, explain_anomaly,
)
from llm.translator import translate


def _safe_list(val):
    if isinstance(val, list): return val
    try: return ast.literal_eval(str(val))
    except: return []


def _tip(term: str, defn: str) -> str:
    """Wrap a financial term in a CSS tooltip span."""
    return f'<span class="tip" data-tip="{defn}">{term}</span>'


# ── Language analysis modal ───────────────────────────────────────────────────

def _build_chart_b64(det_df, ticker: str) -> str:
    """Generate a 90-day price + EMA20 chart and return as base64 PNG string."""
    import io, base64
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    df = det_df.copy().tail(90)
    dates  = df["Date"].values
    closes = df["Close"].values
    ema20  = df["ema_20"].values if "ema_20" in df.columns else None

    fig, ax = plt.subplots(figsize=(10, 3.2))
    fig.patch.set_facecolor("#060a0f")
    ax.set_facecolor("#060a0f")

    # Price area fill
    ax.fill_between(dates, closes, alpha=0.08, color="#1de9b6")
    ax.plot(dates, closes, color="#1de9b6", linewidth=1.6, zorder=3)
    if ema20 is not None:
        ax.plot(dates, ema20, color="#58a6ff", linewidth=1.0, linestyle="--", alpha=0.7, zorder=2)

    # Styling
    for spine in ax.spines.values():
        spine.set_edgecolor("#1a2332")
    ax.tick_params(colors="#3d5266", labelsize=7)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    ax.set_xlim(dates[0], dates[-1])
    ax.grid(axis="y", color="#1a2332", linewidth=0.5, alpha=0.6)
    ax.grid(axis="x", visible=False)
    fig.autofmt_xdate(rotation=0, ha="center")
    plt.tight_layout(pad=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@st.dialog("AI Analyst Report", width="large")
def show_analysis_modal(ticker: str, name: str, det_df, news_df, lang: str) -> None:
    """Show the AI Analyst Report in a large modal with chart and PDF download."""
    st.session_state.lang_modal = None   # reset so closing X doesn't reopen
    from llm.translator import translate
    lang_labels = {"english": "English", "german": "Deutsch", "arabic": "العربية"}

    if news_df is None or ticker not in news_df["ticker"].values:
        st.markdown(
            '<div style="font-size:11px;color:#5c7080;font-family:IBM Plex Mono">'
            'No LLM report available for this ticker.</div>',
            unsafe_allow_html=True,
        )
        return
    llm_text = news_df[news_df["ticker"] == ticker].iloc[0].get("llm_summary", "")
    if not llm_text or len(llm_text) < 120:
        st.markdown(
            '<div style="font-size:11px;color:#5c7080;font-family:IBM Plex Mono">'
            'LLM report not generated yet. Run the narrator pipeline first.</div>',
            unsafe_allow_html=True,
        )
        return

    with st.spinner("Loading…"):
        display_text = translate(llm_text, lang, ticker) if lang != "english" else llm_text

    # ── Build chart base64 ────────────────────────────────────────────────────
    chart_b64   = ""
    chart_tag   = ""
    if det_df is not None and "Close" in det_df.columns and len(det_df) > 5:
        try:
            chart_b64 = _build_chart_b64(det_df, ticker)
            chart_tag = f'<img src="data:image/png;base64,{chart_b64}" style="width:100%;border-radius:6px;margin:20px 0 10px" />'
        except Exception:
            pass

    # ── Build HTML for download ───────────────────────────────────────────────
    rtl_css   = "direction:rtl;text-align:right;" if lang == "arabic" else ""
    safe_body = display_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    html_content = f"""<!DOCTYPE html>
<html lang="{lang[:2]}">
<head>
<meta charset="UTF-8">
<title>FinWatch AI — {name} ({ticker})</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=IBM+Plex+Sans:wght@400;600&display=swap');
  *{{box-sizing:border-box;margin:0;padding:0}}
  html{{background:#060a0f}}
  body{{
    font-family:'IBM Plex Sans',sans-serif;
    max-width:820px;margin:0 auto;padding:48px 36px 64px;
    background:#060a0f;color:#adb5bd;
    line-height:1.9;font-size:13px;
    {rtl_css}
  }}
  .fw-logo{{
    font-family:'IBM Plex Mono',monospace;font-size:13px;font-weight:700;
    background:linear-gradient(135deg,#e2e8f0 0%,#1de9b6 40%,#58a6ff 75%,#a371f7 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;letter-spacing:0.5px;margin-bottom:32px;
    display:inline-block;
  }}
  .badge{{
    display:inline-block;font-family:'IBM Plex Mono',monospace;font-size:9px;
    letter-spacing:3px;text-transform:uppercase;color:#1de9b6;
    border:1px solid rgba(29,233,182,0.3);border-radius:3px;
    padding:3px 10px;margin-bottom:20px;
  }}
  h1{{
    font-family:'IBM Plex Mono',monospace;
    font-size:28px;font-weight:700;letter-spacing:-0.5px;
    color:#e2e8f0;margin-bottom:6px;line-height:1.2;
  }}
  .sub{{font-size:11px;color:#3d5266;font-family:'IBM Plex Mono',monospace;
        letter-spacing:1px;margin-bottom:8px}}
  .divider{{border:none;border-top:1px solid #1a2332;margin:20px 0}}
  p{{margin:10px 0;color:#b8c4ce}}
  strong{{color:#e2e8f0}}
  .footer{{margin-top:40px;font-size:9px;color:#3d5266;
           font-family:'IBM Plex Mono',monospace;letter-spacing:1px;
           border-top:1px solid #1a2332;padding-top:12px}}
</style>
</head>
<body>
<div class="fw-logo">FinWatch AI</div>
<div class="badge">AI · Finance · Risk Assessment</div>
<h1>{name}</h1>
<div class="sub">{ticker} &nbsp;·&nbsp; {lang_labels.get(lang,"English")} &nbsp;·&nbsp; Analyst Report</div>
<hr class="divider">
{chart_tag}
<hr class="divider">
{safe_body}
<div class="footer">Generated by FinWatch AI &nbsp;·&nbsp; For informational purposes only. Not financial advice.</div>
</body></html>"""

    # ── Modal header ──────────────────────────────────────────────────────────
    _m_l, _m_r = st.columns([5, 2])
    with _m_l:
        st.markdown(
            f'<div style="font-size:17px;font-weight:600;color:#e2e8f0;margin-bottom:2px">{name}</div>'
            f'<div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;color:#3d5266;'
            f'font-family:\'IBM Plex Mono\',monospace">{ticker} · {lang_labels.get(lang, lang.upper())}</div>',
            unsafe_allow_html=True,
        )
    with _m_r:
        st.download_button(
            label="⬇ Download PDF",
            data=html_content.encode("utf-8"),
            file_name=f"finwatch_{ticker}_{lang[:2].upper()}.html",
            mime="text/html",
            use_container_width=True,
        )

    st.markdown('<div style="border-top:1px solid #1a2332;margin:12px 0 8px"></div>', unsafe_allow_html=True)

    # ── Chart preview in modal ────────────────────────────────────────────────
    if chart_b64:
        st.markdown(
            f'<img src="data:image/png;base64,{chart_b64}" '
            f'style="width:100%;border-radius:6px;margin-bottom:16px" />',
            unsafe_allow_html=True,
        )
        st.markdown('<div style="border-top:1px solid #1a2332;margin:0 0 16px"></div>', unsafe_allow_html=True)

    # ── Report body ───────────────────────────────────────────────────────────
    rtl      = "direction:rtl;text-align:right;" if lang == "arabic" else ""
    safe_txt = _md_to_html(display_text)
    st.markdown(
        f'<div style="font-size:12px;color:#b8c4ce;line-height:1.9;'
        f'font-family:\'IBM Plex Sans\',sans-serif;{rtl}">'
        f'{safe_txt}</div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(decisions, price_data) -> None:
    """Render sidebar: logo, sector selector, stock list."""
    with st.sidebar:
        st.markdown('<div class="logo">Fin<span>Watch</span> AI</div>', unsafe_allow_html=True)

        if st.button("← Home", key="sb_home", use_container_width=True):
            st.session_state.page = "landing"
            st.rerun()

        sector_names = list(SECTORS.keys())
        cur_sector   = st.session_state.selected_sector
        if cur_sector not in sector_names:
            cur_sector = sector_names[0]
        chosen = st.selectbox("", sector_names,
                              index=sector_names.index(cur_sector),
                              label_visibility="collapsed",
                              key="sector_select")
        if chosen != st.session_state.selected_sector:
            st.session_state.selected_sector = chosen

        st.markdown("<hr style='border-color:#1a2332;margin:4px 0 4px'>", unsafe_allow_html=True)

        for t in SECTORS[chosen]:
            row_s = decisions[decisions["ticker"] == t]
            if row_s.empty:
                continue
            name   = COMPANY_NAMES.get(t, t)
            active = "★ " if t == st.session_state.selected else ""
            btn_col, chg_col = st.columns([7, 3])
            with btn_col:
                if st.button(f"{active}{name} ({t})", key=f"stock_{t}", use_container_width=True):
                    st.session_state.selected     = t
                    st.session_state.anomaly_date = None
                    st.session_state.clicked_date = None
                    st.rerun()
            with chg_col:
                if t in price_data:
                    _, pct = price_data[t]
                    arrow  = "▲" if pct > 0 else ("▼" if pct < 0 else "—")
                    color  = "#1de9b6" if pct > 0 else ("#f85149" if pct < 0 else "#3d5266")
                    st.markdown(
                        f'<div style="font-family:IBM Plex Mono;font-size:11px;color:{color};'
                        f'text-align:right;line-height:1.8;padding-top:3px">'
                        f'{arrow}{abs(pct):.1f}%</div>',
                        unsafe_allow_html=True,
                    )



# ── Stock header ──────────────────────────────────────────────────────────────

def render_stock_header(ticker: str, name: str, det_df, lang: str = "english") -> tuple:
    """Render the stock name / price header + compact language switcher. Returns (last_price, last_chg)."""
    last_price, last_chg = None, None
    if det_df is not None and "Close" in det_df.columns and len(det_df) > 1:
        last_price = det_df["Close"].iloc[-1]
        last_chg   = (det_df["Close"].iloc[-1] - det_df["Close"].iloc[-2]) / det_df["Close"].iloc[-2] * 100

    chg_html   = ""
    price_html = ""
    if last_chg is not None:
        chg_color = "#1de9b6" if last_chg >= 0 else "#f85149"
        chg_arr   = "▲" if last_chg >= 0 else "▼"
        chg_html  = (f'<div style="font-size:12px;color:{chg_color};font-family:\'IBM Plex Mono\',monospace;'
                     f'text-align:right;text-shadow:0 0 12px {chg_color}66">'
                     f'{chg_arr} {abs(last_chg):.2f}%</div>')
    if last_price:
        price_html = (f'<div style="font-size:22px;font-weight:500;color:#e2e8f0;'
                      f'font-family:\'IBM Plex Mono\',monospace;text-align:right">'
                      f'${last_price:.2f}</div>')

    _hl, _hr = st.columns([6, 4])
    with _hl:
        st.markdown(
            f'<div style="padding:6px 0 6px">'
            f'<div class="sh-name">{name}</div>'
            f'<div class="sh-sub">{ticker} · NASDAQ</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with _hr:
        # ── REPORT — prominent, top ───────────────────────────────────────────
        st.markdown(
            '<div style="font-size:21px;font-weight:700;color:#e2e8f0;font-family:\'IBM Plex Mono\',monospace;'
            'padding-top:3px;letter-spacing:4px;text-transform:uppercase;text-align:right;'
            'text-shadow:0 0 18px rgba(226,232,240,0.45)">REPORT</div>',
            unsafe_allow_html=True,
        )
        # ── Language switcher — small, right-aligned, below REPORT ───────────
        st.markdown("""<style>
[data-testid="stMarkdownContainer"]:has(#hdr-lang-marker) ~ [data-testid="stHorizontalBlock"] [data-testid="baseButton-secondary"],
[data-testid="stMarkdownContainer"]:has(#hdr-lang-marker) ~ [data-testid="stHorizontalBlock"] [data-testid="baseButton-primary"] {
    font-size: 8px !important;
    padding: 1px 5px !important;
    min-height: 20px !important;
    line-height: 1.2 !important;
    background-color: #0d1117 !important;
    color: #5a6270 !important;
    border-color: #1e2228 !important;
    border-radius: 3px !important;
    letter-spacing: 1.5px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    box-shadow: none !important;
}
[data-testid="stMarkdownContainer"]:has(#hdr-lang-marker) ~ [data-testid="stHorizontalBlock"] [data-testid="baseButton-primary"] {
    background-color: #161b22 !important;
    color: #8892a4 !important;
    border-color: #30363d !important;
}
</style><div id="hdr-lang-marker"></div>""", unsafe_allow_html=True)
        _lpad, _lc1, _lc2, _lc3 = st.columns([4, 0.8, 0.8, 0.8])
        for _col, (_code, _label) in zip([_lc1, _lc2, _lc3], [("english","en"), ("german","de"), ("arabic","ar")]):
            with _col:
                _active = (lang == _code)
                if st.button(_label, key=f"lang_hdr_{_code}", type="primary" if _active else "secondary", use_container_width=True):
                    st.session_state.language   = _code
                    st.session_state.lang_modal = _code
                    st.rerun()
        # ── Price — below lang buttons ────────────────────────────────────────
        st.markdown(
            f'<div style="padding:4px 0 2px">{price_html}{chg_html}</div>',
            unsafe_allow_html=True,
        )

    return last_price, last_chg


def render_spx_header(spx_df) -> None:
    """Render the S&P 500 header."""
    spx_last, spx_chg = None, None
    if spx_df is not None and len(spx_df) > 1:
        spx_last = spx_df["Close"].iloc[-1]
        spx_chg  = (spx_df["Close"].iloc[-1] - spx_df["Close"].iloc[-2]) / spx_df["Close"].iloc[-2] * 100

    chg_html   = ""
    price_html = f'<span class="sh-price">{spx_last:,.2f}</span>' if spx_last else ""
    if spx_chg is not None:
        chg_html = (f'<span class="sh-up">▲ +{spx_chg:.2f}%</span>'
                    if spx_chg >= 0 else
                    f'<span class="sh-dn">▼ {spx_chg:.2f}%</span>')

    st.markdown(f"""
    <div class="stock-header">
      <span class="sh-name">S&P 500</span>
      <span class="sh-sub">^GSPC · Market Index</span>
      {price_html}
      {chg_html}
    </div>""", unsafe_allow_html=True)


# ── Risk / Summary / News row ─────────────────────────────────────────────────

def render_risk_news_row(ticker: str, row, det_df, decisions, news_df) -> None:
    """Render the 3-column row: AI Risk Analysis | AI Summary | News Sentiment."""
    sev = row["severity"]
    _rc1, _rc2, _rc3 = st.columns(3, gap="small")

    with _rc1:
        sev_cls = SEV_CLS.get(sev, "s-normal")
        mom     = row.get("momentum_signal", "neutral")
        mom_arr   = "▲" if mom == "rising" else "▼" if mom == "falling" else "—"
        mom_cls   = "rp-up" if mom == "rising" else "rp-dn" if mom == "falling" else "rp-val"
        direction = row.get("direction", "stable") if "direction" in row.index else "stable"
        p_down    = row.get("p_down", 0) if "p_down" in row.index else 0
        dir_arr   = "▼" if direction == "down" else "▲" if direction == "up" else "—"
        dir_cls   = "rp-dn" if direction == "down" else "rp-up" if direction == "up" else "rp-val"
        drawdown, es_ratio = None, None
        if det_df is not None:
            if "max_drawdown_30d" in det_df.columns:
                v = det_df["max_drawdown_30d"].iloc[-1]
                if not pd.isna(v): drawdown = v
            if "es_ratio" in det_df.columns:
                v = det_df["es_ratio"].iloc[-1]
                if not pd.isna(v): es_ratio = v
        dd_html = (f'<span class="rp-val rp-dn">{drawdown*100:.1f}%</span>'
                   if drawdown else '<span class="rp-val">—</span>')
        es_html = (f'<span class="rp-val rp-dn">{es_ratio:.2f}</span>'
                   if es_ratio else '<span class="rp-val">—</span>')

        # Anomaly detection row — simple language with historical precision
        anom_count  = 0
        anom_weighted = 0.0
        if det_df is not None and not det_df.empty:
            _last = det_df.iloc[-1]
            for _c in ["z_anomaly", "z_anomaly_60", "if_anomaly", "ae_anomaly"]:
                if _c in _last.index and bool(_last.get(_c)):
                    anom_count += 1
            if "anomaly_score_weighted" in _last.index:
                anom_weighted = float(_last.get("anomaly_score_weighted", 0))

        anom_row_html = ""
        if anom_count > 0:
            # Look up historical precision for the current weighted score
            prec_pct = None
            anom_prec_df = load_anomaly_precision()
            if anom_prec_df is not None:
                candidates = anom_prec_df[
                    (anom_prec_df["detector"] == "anomaly_score_weighted") &
                    (anom_prec_df["signal"].str.extract(r">=\s*([\d.]+)")[0].astype(float) <= anom_weighted)
                ]
                if not candidates.empty:
                    # Pick the highest threshold that still applies
                    prec_pct = int(candidates.sort_values("precision", ascending=False).iloc[0]["precision"] * 100)

            if prec_pct is not None:
                label = (
                    f"Unusual activity detected — in similar past situations, "
                    f"there was a real risk <b>{prec_pct}% of the time</b>"
                )
            else:
                label = f"Unusual activity detected by {anom_count} of 4 sensors"

            anom_row_html = (
                f'<div style="margin:6px 0 4px;padding:6px 8px;'
                f'background:rgba(227,179,65,0.07);border-left:2px solid rgba(227,179,65,0.4);'
                f'border-radius:0 4px 4px 0">'
                f'<span style="font-size:10px;color:#c9a227;line-height:1.5">{label}</span>'
                f'</div>'
            )

        # ── Trading Signal Badge ──────────────────────────────────────────────
        trading_signal = str(row.get("trading_signal", "NEUTRAL")) if "trading_signal" in row.index else "NEUTRAL"
        _sig_cfg = {
            "ENTRY":   ("#1de9b6", "rgba(29,233,182,0.12)", "rgba(29,233,182,0.35)", "▲ ENTRY"),
            "EXIT":    ("#f85149", "rgba(248,81,73,0.12)",  "rgba(248,81,73,0.35)",  "▼ EXIT"),
            "HOLD":    ("#58a6ff", "rgba(88,166,255,0.10)", "rgba(88,166,255,0.30)", "◆ HOLD"),
            "NEUTRAL": ("#637a91", "rgba(30,45,65,0.4)",    "rgba(30,45,65,0.6)",   "— NEUTRAL"),
        }
        sig_c, sig_bg, sig_border, sig_label = _sig_cfg.get(trading_signal, _sig_cfg["NEUTRAL"])
        signal_html = f"""
        <div style="display:inline-flex;align-items:center;gap:8px;margin:6px 0 8px">
          <div style="background:{sig_bg};border:1px solid {sig_border};
                      border-radius:6px;padding:5px 14px;
                      box-shadow:0 0 12px {sig_bg}">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:13px;
                         font-weight:700;color:{sig_c};letter-spacing:1px">{sig_label}</span>
          </div>
          <span style="font-size:9px;color:#3d5266;font-family:'IBM Plex Mono',monospace">AI SIGNAL</span>
        </div>"""

        p_drawdown = row.get("p_drawdown", None)
        dd_prob_html = ""
        if p_drawdown is not None:
            try:
                p_val = float(p_drawdown)
                p_col = "#f85149" if p_val >= 0.5 else "#e3b341" if p_val >= 0.35 else "#1de9b6"
                dd_prob_html = f'<div class="rp-row"><span class="rp-key">Drawdown Prob</span><span class="rp-val" style="color:{p_col}">{p_val*100:.0f}%</span></div>'
            except (TypeError, ValueError):
                pass

        st.markdown(f"""
        <div class="rp-section">
          <div class="rp-title">AI Risk Analysis</div>
          <div class="sev-big {sev_cls}">{sev.replace('_',' ')}</div>
          <div style="font-size:9px;color:#5c7080;margin-bottom:4px;font-family:IBM Plex Mono">{row.get('action','—')} · {row.get('date','—')}</div>
          {signal_html}
          {anom_row_html}
          {dd_prob_html}
          <div class="rp-row"><span class="rp-key">Momentum</span>
            <span class="rp-val {mom_cls}">{mom_arr} {mom}</span></div>
          <div class="rp-row"><span class="rp-key">ES Ratio</span>{es_html}</div>
          <div class="rp-row"><span class="rp-key">Drawdown 30D</span>{dd_html}</div>
        </div>""", unsafe_allow_html=True)

    with _rc2:
        summary = row.get("summary", "")
        caution = row.get("caution_flag", None)
        if summary:
            caution_html = (
                f'<div style="color:#e3b341;font-size:10px;margin-top:6px;'
                f'font-family:IBM Plex Mono">⚠ {caution}</div>'
                if caution else ""
            )
            st.markdown(f"""
            <div class="rp-section">
              <div class="rp-title">AI Summary</div>
              <div class="summary-text">{summary}{caution_html}</div>
            </div>""", unsafe_allow_html=True)

    with _rc3:
        if news_df is not None and ticker in news_df["ticker"].values:
            nr         = news_df[news_df["ticker"] == ticker].iloc[0]
            headlines  = _safe_list(nr.get("top_news", []))
            sentiments = _safe_list(nr.get("news_sentiment", []))
            sources    = _safe_list(nr.get("news_sources", nr.get("news_urls", [])))
            if headlines:
                news_html = ""
                for i, h in enumerate(headlines[:3]):
                    sent    = sentiments[i] if i < len(sentiments) else "neutral"
                    s_cls   = {"positive": "s-pos", "negative": "s-neg", "neutral": "s-neu"}.get(sent, "s-neu")
                    source  = sources[i] if i < len(sources) and sources[i] else ""
                    if source.startswith("http"):
                        source = urllib.parse.urlparse(source).netloc.replace("www.", "") or ""
                    source_html = (
                        f'<span style="font-size:9px;color:#5c7080;font-family:IBM Plex Mono;'
                        f'display:inline-block;margin-top:3px">Source: {source}</span>'
                    ) if source else ""
                    news_html += (
                        f'<div class="news-item">'
                        f'<span class="news-sent {s_cls}">{sent.upper()}</span>'
                        f'<div class="news-text">{h}</div>'
                        f'{source_html}</div>'
                    )
                st.markdown(
                    f'<div class="rp-section"><div class="rp-title">News Sentiment</div>'
                    f'{news_html}</div>',
                    unsafe_allow_html=True,
                )


# ── Anomaly selector ──────────────────────────────────────────────────────────

_ANOMALY_PERIOD_OFFSETS = {
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6),
    "1Y": pd.DateOffset(years=1),
    "All": None,
}
_ANOMALY_DET_COLS = ["z_anomaly", "z_anomaly_60", "if_anomaly", "ae_anomaly"]


def render_anomaly_selector(det_df, name: str, period: str = "1M", dec_row=None) -> None:
    """Render the anomaly date dropdown filtered to the active chart period."""
    plot_df = det_df.copy()

    # Filter to active period
    available = [c for c in _ANOMALY_DET_COLS if c in plot_df.columns]
    if not available:
        return

    x_end  = plot_df["Date"].iloc[-1]
    offset = _ANOMALY_PERIOD_OFFSETS.get(period)
    if offset is not None:
        plot_df = plot_df[plot_df["Date"] >= (x_end - offset)]

    # Only dates where ≥2 detectors fired
    confirmed = plot_df[available].apply(
        lambda row: sum(bool(v) for v in row), axis=1
    )
    anom_dates = (
        plot_df[confirmed >= 2]["Date"]
        .dt.strftime("%Y-%m-%d")
        .tolist()
    )
    if not anom_dates:
        return

    # Reset selected date if it's outside the current period
    if st.session_state.anomaly_date not in anom_dates:
        st.session_state.anomaly_date = None

    options = ["— select anomaly date —"] + anom_dates
    cur_idx = 0
    if st.session_state.anomaly_date in anom_dates:
        cur_idx = anom_dates.index(st.session_state.anomaly_date) + 1
    chosen = st.selectbox("Anomaly", options, index=cur_idx,
                          label_visibility="collapsed", key="anomaly_select")
    st.session_state.anomaly_date = chosen if chosen != "— select anomaly date —" else None

    if st.session_state.anomaly_date:
        adate   = pd.Timestamp(st.session_state.anomaly_date)
        arow_df = det_df[det_df["Date"].dt.date == adate.date()]
        if not arow_df.empty:
            _dec = dec_row.iloc[0] if dec_row is not None and not dec_row.empty else None
            explanation = explain_anomaly(arow_df.iloc[0], name, st.session_state.anomaly_date, _dec)
            if explanation:
                st.markdown(explanation, unsafe_allow_html=True)


# ── Candle detail panel ───────────────────────────────────────────────────────

def render_candle_panel(det_df, clicked_date: str) -> None:
    """Horizontal strip below the chart showing candle details on click."""
    if clicked_date is None or det_df is None:
        return

    date_ts = pd.Timestamp(clicked_date)
    row_df  = det_df[det_df["Date"].dt.date == date_ts.date()]
    if row_df.empty:
        return
    r = row_df.iloc[0]

    close_v = r.get("Close", None)
    open_v  = r.get("Open", None)
    high_v  = r.get("High", None)
    low_v   = r.get("Low", None)
    rsi_v   = r.get("rsi", None)
    vol_v   = r.get("Volume", None)
    ret_v   = r.get("returns", None)

    idx     = row_df.index[0]
    ema20_v = det_df.loc[:idx, "Close"].ewm(span=20).mean().iloc[-1]

    _det_map = [("z_anomaly","Z-30D"),("z_anomaly_60","Z-60D"),
                ("if_anomaly","IsoForest"),("ae_anomaly","LSTM-AE")]
    fired_labels = [lbl for col, lbl in _det_map if bool(r.get(col, False))]

    _TIPS = {
        "O":       "Open — price at market open",
        "H":       "High — highest price of the day",
        "L":       "Low — lowest price of the day",
        "C":       "Close — price at market close",
        "Close":   "Close — price at market close",
        "EMA 20":  "EMA 20 — Exponential Moving Average over 20 days. Smoothed trend line; price above = bullish.",
        "Δ 1D":    "Daily change — % price change vs previous close",
        "RSI":     "RSI — Relative Strength Index (0–100). Above 70 = overbought, below 30 = oversold.",
        "Vol":     "Volume — number of shares traded on this day",
        "ANOMALY": "Anomaly — models that flagged unusual price or volume behavior on this day",
    }

    def _cell(label, value, color="#8b949e"):
        tip = _TIPS.get(label, "")
        tip_attr = f' data-tip="{tip}"' if tip else ""
        return (
            '<div style="display:flex;flex-direction:column;align-items:center;'
            'padding:0 12px;border-right:1px solid #1a2332">'
            f'<span style="font-size:8px;color:#5c7080;font-family:IBM Plex Mono;'
            f'letter-spacing:1px;margin-bottom:2px;cursor:help"{tip_attr}>{label}</span>'
            f'<span style="font-size:11px;color:{color};font-family:IBM Plex Mono;'
            f'font-weight:500">{value}</span>'
            '</div>'
        )

    ret_color = "#1de9b6" if (close_v or 0) >= (open_v or close_v or 0) else "#f85149"

    cells = f'<div style="font-size:10px;color:#cdd9e5;font-family:IBM Plex Mono;font-weight:500;padding:0 12px;border-right:1px solid #1a2332;align-self:center">{clicked_date}</div>'

    if open_v is not None:
        cells += _cell("O", f"${open_v:,.2f}")
        cells += _cell("H", f"${high_v:,.2f}", "#1de9b6")
        cells += _cell("L", f"${low_v:,.2f}", "#f85149")
        cells += _cell("C", f"${close_v:,.2f}", ret_color)
    elif close_v is not None:
        cells += _cell("Close", f"${close_v:,.2f}", ret_color)

    cells += _cell("EMA 20", f"${ema20_v:,.2f}", "#a371f7")

    if ret_v is not None:
        rc  = "#1de9b6" if ret_v >= 0 else "#f85149"
        sym = "▲" if ret_v >= 0 else "▼"
        cells += _cell("Δ 1D", f"{sym}{abs(ret_v)*100:.2f}%", rc)
    if rsi_v is not None:
        rc = "#f85149" if rsi_v > 70 else "#3fb950" if rsi_v < 30 else "#8b949e"
        cells += _cell("RSI", f"{rsi_v:.1f}", rc)
    if vol_v is not None:
        vf = f"{vol_v/1e6:.1f}M" if vol_v >= 1e6 else f"{vol_v/1e3:.0f}K"
        cells += _cell("Vol", vf)

    if fired_labels:
        anom_txt = " · ".join(fired_labels)
        tip = _TIPS["ANOMALY"]
        cells += (
            '<div style="display:flex;flex-direction:column;align-items:center;'
            'padding:0 12px;border-right:1px solid #1a2332">'
            f'<span style="font-size:8px;color:#5c7080;font-family:IBM Plex Mono;'
            f'letter-spacing:1px;margin-bottom:2px;cursor:help" data-tip="{tip}">ANOMALY</span>'
            f'<span style="font-size:10px;color:#e3b341;font-family:IBM Plex Mono">{anom_txt}</span>'
            '</div>'
        )

    _sc, _xc = st.columns([50, 1], gap="small")
    with _sc:
        st.markdown(
            '<div style="background:#0d1117;border:1px solid #1a2332;border-radius:3px;'
            'padding:6px 4px;display:flex;flex-direction:row;align-items:stretch;overflow-x:auto">'
            f'{cells}'
            '</div>',
            unsafe_allow_html=True,
        )
    with _xc:
        if st.button("✕", key="clear_candle", use_container_width=True,
                     help="Clear selection"):
            st.session_state.clicked_date = None
            st.rerun()


# ── Analysis panel + LLM report ───────────────────────────────────────────────

def render_strategy_box(det_df, dec_row) -> None:
    """AI Trading Signal (ML) + rule-based context."""
    if det_df is None or det_df.empty or dec_row is None or dec_row.empty:
        return

    last      = det_df.iloc[-1]
    row       = dec_row.iloc[0]
    close     = float(last.get("Close", 0))
    ema20     = float(det_df["Close"].ewm(span=20).mean().iloc[-1])
    rsi       = float(last.get("rsi", 50))
    mom_sig   = str(row.get("momentum_signal", "neutral"))
    sev       = str(row.get("severity", "NORMAL"))
    p_drawdown = float(row.get("p_drawdown", 0.33)) if row.get("p_drawdown") is not None else 0.33
    pe_ratio  = row.get("pe_ratio", None)
    revenue_growth = row.get("revenue_growth", None)

    above_ema = close > ema20
    confirmed_det = [c for c in ["z_anomaly","z_anomaly_60","if_anomaly","ae_anomaly"]
                     if c in last.index and bool(last.get(c))]
    n_anom = len(confirmed_det)

    # ── ML Trading Signal (primary) ───────────────────────────────────────────
    trading_signal = str(row.get("trading_signal", "NEUTRAL")) if "trading_signal" in row.index else "NEUTRAL"
    confidence     = float(row.get("confidence", 0)) if row.get("confidence") is not None else 0.0

    _sig_cfg = {
        "ENTRY":   ("#1de9b6", "rgba(29,233,182,0.10)", "rgba(29,233,182,0.30)", "▲ ENTRY"),
        "EXIT":    ("#f85149", "rgba(248,81,73,0.10)",  "rgba(248,81,73,0.30)",  "▼ EXIT"),
        "HOLD":    ("#58a6ff", "rgba(88,166,255,0.08)", "rgba(88,166,255,0.25)", "◆ HOLD"),
        "NEUTRAL": ("#637a91", "rgba(30,45,65,0.3)",    "rgba(30,45,65,0.5)",   "— NEUTRAL"),
    }
    sig_c, sig_bg, sig_border, sig_label = _sig_cfg.get(trading_signal, _sig_cfg["NEUTRAL"])

    _sig_desc = {
        "ENTRY": "Models indicate low drawdown risk, positive momentum, and the stock is above its MA200. Conditions are favourable for a new position.",
        "EXIT":  "High drawdown probability or critical anomaly detected. Models recommend reducing or closing the position.",
        "HOLD":  "Mixed or neutral signals. No strong case for entry or exit at this time — maintain current position and monitor.",
        "NEUTRAL": "Conflicting signals. No clear directional edge. Wait for confirmation.",
    }
    sig_desc = _sig_desc.get(trading_signal, "")

    conf_html = f'<span style="font-size:12px;color:#3d5266;font-family:IBM Plex Mono;margin-left:10px">Confidence: {confidence*100:.0f}%</span>' if confidence > 0 else ""

    # ── Rule-based context (secondary, compact) ────────────────────────────────
    _ema_tip  = _tip("EMA20", "20-day Exponential Moving Average — short-term trend line.")
    _rsi_tip  = _tip("RSI", "Relative Strength Index. Above 70: overbought. Below 30: oversold.")
    _mom_tip  = _tip("Momentum", "Speed and direction of recent price change.")
    _dd_tip   = _tip("Drawdown prob", "AI probability of a >5% price drop in the next 10 days.")
    _anom_tip = _tip("anomaly detectors", "AI models that flag unusual price or volume behavior.")
    _pe_tip   = _tip("P/E", "Price-to-Earnings ratio — how much investors pay per $1 of earnings.")
    _rev_tip  = _tip("Revenue growth", "Year-over-year change in company revenue.")

    context_items = []
    context_items.append((f"above {_ema_tip}" if above_ema else f"below {_ema_tip}",
                           "#1de9b6" if above_ema else "#f85149"))
    context_items.append((f"{_rsi_tip} {rsi:.0f}" + (" — overbought" if rsi > 70 else " — oversold" if rsi < 30 else ""),
                           "#f85149" if rsi > 70 else "#1de9b6" if rsi < 30 else "#8b949e"))
    context_items.append((f"{_mom_tip}: {mom_sig}",
                           "#1de9b6" if mom_sig == "rising" else "#f85149" if mom_sig == "falling" else "#8b949e"))
    context_items.append((f"{_dd_tip}: {p_drawdown*100:.0f}%",
                           "#f85149" if p_drawdown >= 0.5 else "#e3b341" if p_drawdown >= 0.35 else "#1de9b6"))
    if n_anom > 0:
        context_items.append((f"{n_anom}/4 {_anom_tip} triggered", "#e3b341"))
    if pe_ratio is not None:
        try:
            pe_val = float(pe_ratio)
            if pe_val < 0:
                context_items.append((f"{_pe_tip} negative — earnings loss", "#f85149"))
            elif pe_val > 50:
                context_items.append((f"{_pe_tip} {pe_val:.0f} — elevated valuation", "#e3b341"))
            else:
                context_items.append((f"{_pe_tip} {pe_val:.0f}", "#8b949e"))
        except (TypeError, ValueError):
            pass
    if revenue_growth is not None:
        try:
            rg = float(revenue_growth)
            if rg < -0.05:
                context_items.append((f"{_rev_tip} {rg*100:.0f}%", "#f85149"))
            elif rg > 0.10:
                context_items.append((f"{_rev_tip} +{rg*100:.0f}%", "#1de9b6"))
        except (TypeError, ValueError):
            pass

    ctx_html = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;padding:4px 0">'
        f'<span style="color:{c};font-size:11px;flex-shrink:0">●</span>'
        f'<span style="font-size:10px;color:#b8c4ce;line-height:1.4">{lbl}</span>'
        f'</div>'
        for lbl, c in context_items
    )

    st.markdown(f"""
    <div style="background:rgba(11,17,26,0.8);border:1px solid rgba(30,45,65,0.6);
                border-radius:10px;padding:16px 20px;margin-top:10px;
                box-shadow:0 4px 20px rgba(0,0,0,0.3),inset 0 1px 0 rgba(255,255,255,0.03)">
      <div style="font-size:10px;letter-spacing:2.5px;text-transform:uppercase;
                  color:#3d5266;font-family:'IBM Plex Mono',monospace;
                  border-bottom:1px solid rgba(30,45,65,0.6);padding-bottom:8px;margin-bottom:14px">
        AI Trading Signal
      </div>
      <div style="display:grid;grid-template-columns:auto 1fr;gap:16px;align-items:start">
        <div>
          <div style="background:{sig_bg};border:1px solid {sig_border};border-radius:8px;
                      padding:10px 22px;text-align:center;
                      box-shadow:0 0 20px {sig_bg}">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:20px;font-weight:700;
                        color:{sig_c};letter-spacing:2px">{sig_label}</div>
          </div>
          {conf_html}
        </div>
        <div>
          <div style="font-size:11px;color:#b8c4ce;line-height:1.6;margin-bottom:10px">{sig_desc}</div>
          <div style="border-top:1px solid rgba(30,45,65,0.5);padding-top:8px">
            <div style="font-size:10px;letter-spacing:1.5px;color:#3d5266;
                        font-family:'IBM Plex Mono',monospace;text-transform:uppercase;
                        margin-bottom:6px">Signal Factors</div>
            {ctx_html}
          </div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


def render_analysis_panel(det_df, dec_row, ticker: str, name: str, lang: str) -> None:
    """Render the 4-column HTML analysis grid + risk drivers as separate elements."""
    result = build_analysis(det_df, dec_row, ticker, name, lang)
    if result is None:
        return
    main_html, drivers_html = result
    st.markdown(main_html, unsafe_allow_html=True)
    if drivers_html:
        st.markdown(drivers_html, unsafe_allow_html=True)


def render_investor_summary(det_df, dec_row, row, news_df, ticker: str) -> None:
    """Render the short investor signal summary."""
    html = build_investor_summary(det_df, dec_row, row, news_df, ticker)
    if html:
        st.markdown(html, unsafe_allow_html=True)


def _md_to_html(text: str) -> str:
    """Safely convert markdown text to HTML for embedding inside a div."""
    escaped = _html.escape(text)
    escaped = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', escaped)
    escaped = re.sub(r'^---\s*$', '<hr style="border:none;border-top:1px solid #1a2332;margin:8px 0">', escaped, flags=re.MULTILINE)
    escaped = escaped.replace('\n', '<br>')
    return escaped


def render_llm_report(ticker: str, news_df, lang: str, dec_row=None) -> None:
    """Render the full LLM analyst report with integrated language switcher."""
    llm_text  = ""
    fallback  = False

    if news_df is not None and ticker in news_df["ticker"].values:
        llm_text = news_df[news_df["ticker"] == ticker].iloc[0].get("llm_summary", "") or ""

    if len(llm_text) < 120:
        if dec_row is not None and not dec_row.empty:
            fb = dec_row.iloc[0].get("summary", "") or ""
            if len(fb) > 10:
                llm_text = fb
                fallback = True
        if not fallback:
            return

    # ── Section divider ──────────────────────────────────────────────────────
    st.markdown(
        '<div style="border-top:1px solid #1a2332;margin:20px 0 0"></div>',
        unsafe_allow_html=True,
    )

    # ── Header row: title left, language tabs right ──────────────────────────
    report_label = "AI Risk Summary" if fallback else "AI Analyst Report"
    _hdr_l, _hdr_r = st.columns([5, 3])
    with _hdr_l:
        st.markdown(
            f'<div style="font-size:17px;font-weight:600;color:#e2e8f0;'
            f'font-family:\'Inter\',sans-serif;padding-top:10px;letter-spacing:-0.3px">'
            f'{report_label}</div>',
            unsafe_allow_html=True,
        )
    with _hdr_r:
        st.markdown("""<style>
[data-testid="stMarkdownContainer"]:has(#rpt-lang-marker) ~ [data-testid="stHorizontalBlock"] [data-testid="baseButton-secondary"],
[data-testid="stMarkdownContainer"]:has(#rpt-lang-marker) ~ [data-testid="stHorizontalBlock"] [data-testid="baseButton-primary"] {
    font-size: 8px !important;
    padding: 1px 5px !important;
    min-height: 22px !important;
    line-height: 1.2 !important;
    background-color: #0d1117 !important;
    color: #5a6270 !important;
    border-color: #1e2228 !important;
    border-radius: 3px !important;
    letter-spacing: 1.5px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    box-shadow: none !important;
}
[data-testid="stMarkdownContainer"]:has(#rpt-lang-marker) ~ [data-testid="stHorizontalBlock"] [data-testid="baseButton-primary"] {
    background-color: #161b22 !important;
    color: #8892a4 !important;
    border-color: #30363d !important;
}
</style><div id="rpt-lang-marker"></div>""", unsafe_allow_html=True)
        _lang_cols = st.columns([1, 1, 1, 2])
        for _col, (_code, _label) in zip(_lang_cols[:3], [("english","EN"), ("german","DE"), ("arabic","AR")]):
            with _col:
                _active = (lang == _code)
                if st.button(
                    _label,
                    key=f"lang_{_code}",
                    type="primary" if _active else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.language = _code
                    st.rerun()

    # ── Report body ───────────────────────────────────────────────────────────
    with st.spinner("Translating..." if (lang != "english" and not fallback) else ""):
        display_text = translate(llm_text, lang, ticker) if not fallback else llm_text

    rtl_style = "direction:rtl;text-align:right;" if lang == "arabic" else ""
    note_html = (
        '<div style="font-size:9px;color:#3d5266;font-family:IBM Plex Mono;margin-top:10px">'
        'Full LLM report not generated yet — re-run the narrator pipeline.</div>'
    ) if fallback else ""

    safe_text = _md_to_html(display_text)

    st.markdown(
        f'<div style="background:rgba(13,17,23,0.7);border:1px solid #1a2332;'
        f'border-radius:6px;padding:16px 20px;margin-top:6px">'
        f'<div style="font-size:12px;color:#b8c4ce;line-height:1.9;'
        f'font-family:\'IBM Plex Sans\',sans-serif;{rtl_style}">{safe_text}</div>'
        f'{note_html}</div>',
        unsafe_allow_html=True,
    )
