"""
FinWatch AI — Technical Signal Features
=========================================
Shared feature computation used by:
  - prediction/models/drawdown_probability.py  (training + inference)
  - decision/decision_pipeline.py              (live decision making)

Functions operate on a full DataFrame (all tickers, all dates) and add
signal columns row-by-row per ticker. This ensures training and inference
use identical feature logic.
"""

import pandas as pd


def add_vix(data: pd.DataFrame) -> pd.DataFrame:
    """Fetch VIX from FRED and merge onto data. Falls back to neutral 20."""
    try:
        from pandas_datareader import data as web
        vix = web.DataReader("VIXCLS", "fred", "2015-01-01", "2030-12-31").reset_index()
        vix.columns = ["Date", "vix_level"]
        vix["Date"] = pd.to_datetime(vix["Date"])
    except Exception:
        vix = pd.DataFrame({
            "Date": pd.date_range("2015-01-01", "2030-12-31"),
            "vix_level": 20.0,
        })
    vix["vix_change"] = vix["vix_level"].pct_change(fill_method=None).fillna(0)
    vix["vix_high"]   = (vix["vix_level"] > 25).astype(int)
    data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)
    data = data.merge(vix[["Date", "vix_level", "vix_change", "vix_high"]],
                      on="Date", how="left")
    data["vix_level"]  = data["vix_level"].ffill().fillna(20)
    data["vix_change"] = data["vix_change"].ffill().fillna(0)
    data["vix_high"]   = data["vix_high"].ffill().fillna(0)
    return data


def add_stock_ma_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add each stock's own 50-day and 200-day MA position.
    Note: ma50/ma200 columns in detection parquets are SPX MAs, not stock MAs.
      price_vs_ma50_stock  = (Close / own MA50)  - 1
      price_vs_ma200_stock = (Close / own MA200) - 1
    """
    def _per_ticker(df):
        df    = df.copy().sort_values("Date")
        close = df["Close"]
        ma50  = close.rolling(50,  min_periods=20).mean()
        ma200 = close.rolling(200, min_periods=50).mean()
        df["price_vs_ma50_stock"]  = (close / ma50  - 1).where(ma50  > 0, 0.0)
        df["price_vs_ma200_stock"] = (close / ma200 - 1).where(ma200 > 0, 0.0)
        return df
    return data.groupby("ticker", group_keys=False).apply(_per_ticker)


def add_technical_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute bullish and bearish technical signal features row-by-row per ticker.

    Bullish:
      price_vs_ema20          — (Close / EMA20) - 1: short-term trend position
      ma50_above_ma200        — 1 if MA50 > MA200 (golden cross regime)
      golden_cross            — MA50 crossed above MA200 in last 20 days
      macd_cross_bullish      — MACD crossed above signal line in last 3 days
      macd_hist               — MACD histogram (positive = bullish momentum)
      hh_hl                   — Higher Highs + Higher Lows (uptrend structure)
      rsi_divergence_bullish  — Price lower lows but RSI higher lows
      consolidation_above_ma50 — Tight range above MA50 (bullish coil)

    Bearish:
      death_cross             — MA50 crossed below MA200 in last 20 days
      macd_cross_bearish      — MACD crossed below signal line in last 3 days
      ll_lh                   — Lower Lows + Lower Highs (downtrend structure)
      panic_volume            — Price drop > 2% + volume > 2x average
      volume_spike_no_recovery — Volume spike 2-5 days ago, price still declining
      rsi_below_45_high_vol   — RSI < 45 declining with rising volume

    Neutral:
      vol_ratio               — Current volume / 20-day average (continuous)
      consolidation           — Tight price range < 3% over 10 days
    """
    def _per_ticker(df):
        df     = df.copy().sort_values("Date")
        close  = df["Close"].astype(float)
        high   = df["High"].astype(float)   if "High"   in df.columns else close
        low    = df["Low"].astype(float)    if "Low"    in df.columns else close
        volume = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(0.0, index=df.index)
        rsi_s  = df["rsi"].astype(float)    if "rsi"    in df.columns else pd.Series(50.0, index=df.index)

        # ── MAs and EMA ────────────────────────────────────────────────────────
        ema20 = close.ewm(span=20, adjust=False).mean()
        ma50  = close.rolling(50,  min_periods=20).mean()
        ma200 = close.rolling(200, min_periods=50).mean()

        df["price_vs_ema20"]   = (close / ema20 - 1).where(ema20 > 0, 0.0)
        df["ma50_above_ma200"] = (ma50 > ma200).astype(int)

        # ── Golden Cross / Death Cross (last 20 days) ──────────────────────────
        df["golden_cross"] = ((ma50 > ma200) & (ma50.shift(20) < ma200.shift(20))).astype(int).fillna(0)
        df["death_cross"]  = ((ma50 < ma200) & (ma50.shift(20) > ma200.shift(20))).astype(int).fillna(0)

        # ── MACD ───────────────────────────────────────────────────────────────
        ema12     = close.ewm(span=12, adjust=False).mean()
        ema26     = close.ewm(span=26, adjust=False).mean()
        macd      = ema12 - ema26
        macd_sig  = macd.ewm(span=9, adjust=False).mean()
        macd_3    = macd.shift(3)
        sig_3     = macd_sig.shift(3)
        df["macd_cross_bullish"] = ((macd > macd_sig) & (macd_3 < sig_3)).astype(int).fillna(0)
        df["macd_cross_bearish"] = ((macd < macd_sig) & (macd_3 > sig_3)).astype(int).fillna(0)
        df["macd_hist"]          = (macd - macd_sig).fillna(0)

        # ── Higher Highs + Higher Lows / Lower Lows + Lower Highs ─────────────
        h_curr = high.rolling(10, min_periods=5).max()
        h_prev = high.shift(10).rolling(10, min_periods=5).max()
        l_curr = low.rolling(10, min_periods=5).min()
        l_prev = low.shift(10).rolling(10, min_periods=5).min()
        df["hh_hl"] = ((h_curr > h_prev) & (l_curr > l_prev)).astype(int).fillna(0)
        df["ll_lh"] = ((h_curr < h_prev) & (l_curr < l_prev)).astype(int).fillna(0)

        # ── Volume signals ─────────────────────────────────────────────────────
        vol_ma20  = volume.rolling(20, min_periods=5).mean()
        vol_ratio = (volume / vol_ma20.where(vol_ma20 > 0, 1.0)).fillna(1.0)
        ret       = close.pct_change().fillna(0)

        df["vol_ratio"]    = vol_ratio
        df["panic_volume"] = ((vol_ratio > 2.0) & (ret < -0.02)).astype(int)

        spike_past = vol_ratio.shift(2).rolling(4, min_periods=1).max() > 2.0
        still_down = (close / close.shift(5) - 1) < -0.02
        df["volume_spike_no_recovery"] = (spike_past & still_down).astype(int).fillna(0)

        # High-volume breakout after quiet consolidation period
        prior_vol_avg    = volume.shift(5).rolling(15, min_periods=5).mean()
        consolidation_vol = prior_vol_avg < vol_ma20 * 0.85
        df["volume_breakout"] = (
            (vol_ratio > 1.5) & (ret > 0) & consolidation_vol
        ).astype(int).fillna(0)

        # Low volume: nobody trading (below 70% of 20-day average)
        df["low_volume"] = (volume < vol_ma20 * 0.70).astype(int).fillna(0)

        # ── RSI signals ────────────────────────────────────────────────────────
        df["rsi_below_45_high_vol"] = (
            (rsi_s < 45) & (rsi_s < rsi_s.shift(5)) & (vol_ratio > 1.3)
        ).astype(int).fillna(0)

        # RSI oversold bounce: RSI was < 30 in last 10 days, now recovered above 35
        rsi_min_10 = rsi_s.rolling(10, min_periods=5).min()
        df["rsi_oversold_bounce"]    = ((rsi_min_10 < 30) & (rsi_s > 35)).astype(int).fillna(0)
        # RSI oversold no bounce: still below 35 and declining (no rebound)
        df["rsi_oversold_no_bounce"] = ((rsi_s < 35) & (rsi_s < rsi_s.shift(4))).astype(int).fillna(0)

        # ── RSI divergence: price lower lows, RSI higher lows ─────────────────
        pl_curr = close.rolling(10, min_periods=5).min()
        pl_prev = close.shift(10).rolling(10, min_periods=5).min()
        rl_curr = rsi_s.rolling(10, min_periods=5).min()
        rl_prev = rsi_s.shift(10).rolling(10, min_periods=5).min()
        df["rsi_divergence_bullish"] = (
            (pl_curr < pl_prev) & (rl_curr > rl_prev)
        ).astype(int).fillna(0)

        # ── Consolidation ──────────────────────────────────────────────────────
        roll_max  = close.rolling(10, min_periods=5).max()
        roll_min  = close.rolling(10, min_periods=5).min()
        roll_mean = close.rolling(10, min_periods=5).mean()
        range_pct = (roll_max - roll_min) / roll_mean.where(roll_mean > 0, 1.0)
        df["consolidation"]           = (range_pct < 0.03).astype(int).fillna(0)
        df["consolidation_above_ma50"] = (
            (range_pct < 0.03) & (close > ma50)
        ).astype(int).fillna(0)

        return df

    return data.groupby("ticker", group_keys=False).apply(_per_ticker)


# Columns added by add_technical_signals() — used to extract latest values per ticker
TECHNICAL_SIGNAL_COLS = [
    "price_vs_ema20", "ma50_above_ma200",
    "golden_cross", "death_cross",
    "macd_cross_bullish", "macd_cross_bearish", "macd_hist",
    "hh_hl", "ll_lh",
    "panic_volume", "vol_ratio", "volume_spike_no_recovery",
    "volume_breakout", "low_volume",
    "rsi_below_45_high_vol", "rsi_oversold_bounce", "rsi_oversold_no_bounce",
    "rsi_divergence_bullish",
    "consolidation", "consolidation_above_ma50",
]
