"""
FinWatch AI — Layer 7B: Narrative Engine
==========================================
Takes Decision + SHAP top-3 + OBV signal and produces a structured narrative.

4 Rules:
  1. signals_aligned_bearish  — CRITICAL/WARNING + OBV negative + driver pushes risk up
  2. signals_aligned_bullish  — NORMAL/WATCH + OBV positive + driver pushes risk down
  3. conflict_dead_cat_bounce — CRITICAL/WARNING + OBV strongly positive
  4. blind_spot_review        — NORMAL/WATCH + OBV strongly negative
  5. mixed                    — no dominant pattern

Output per ticker:
  driver        — top-1 SHAP feature name
  driver_shap   — SHAP value (+ = risk up, - = risk down)
  confirmation  — sell_pressure / buy_pressure / neutral
  conflict      — obv_contradicts_severity | obv_signals_hidden_risk | ""
  narrative     — narrative key (one of the 5 above)
  narrative_text — human-readable sentence (German)
"""

# OBV thresholds
# obv_signal = volume_zscore * returns → typical range: -0.20 to +0.20
OBV_SELL_WEAK   = -0.02   # mild selling pressure
OBV_SELL_STRONG = -0.08   # strong selling pressure (blind spot trigger)
OBV_BUY_WEAK    =  0.02   # mild buying pressure
OBV_BUY_STRONG  =  0.04   # strong buying pressure (conflict trigger)

BEARISH_SEVERITIES = {"CRITICAL", "WARNING"}
BULLISH_SEVERITIES = {"NORMAL", "WATCH"}


def obv_label(obv: float) -> str:
    if obv < OBV_SELL_WEAK:
        return "sell_pressure"
    if obv > OBV_BUY_WEAK:
        return "buy_pressure"
    return "neutral"


def detect_conflict(severity: str, obv: float) -> str:
    """Returns conflict description or empty string."""
    if severity in BEARISH_SEVERITIES and obv > OBV_BUY_STRONG:
        return "obv_contradicts_severity"
    if severity in BULLISH_SEVERITIES and obv < OBV_SELL_STRONG:
        return "obv_signals_hidden_risk"
    return ""


def build(severity: str, obv: float, driver_shap: float) -> tuple[str, str]:
    """
    Returns (narrative_key, narrative_text).

    driver_shap is available for context but NOT used as a rule condition —
    it is already captured in the 'driver' and 'driver_shap' output columns.
    Note: CRITICAL can be triggered by Drawdown Override even when XGBoost
    predicts low risk (negative SHAP), so we must not require driver_shap > 0.
    """
    # Rule 1: severity bearish + OBV confirms selling pressure
    if severity in BEARISH_SEVERITIES and obv < OBV_SELL_WEAK:
        return (
            "signals_aligned_bearish",
            "Alle Signale zeigen in dieselbe Richtung — höchste Priorität.",
        )

    # Rule 2: severity bullish + OBV confirms buying pressure
    if severity in BULLISH_SEVERITIES and obv > OBV_BUY_WEAK:
        return (
            "signals_aligned_bullish",
            "Starker Aufwärtstrend bestätigt durch Kaufdruck.",
        )

    # Rule 3: conflict — dangerous
    if severity in BEARISH_SEVERITIES and obv > OBV_BUY_STRONG:
        return (
            "conflict_dead_cat_bounce",
            "Modell warnt, aber Kaufdruck erkannt — möglicher Dead Cat Bounce.",
        )

    # Rule 4: blind spot
    if severity in BULLISH_SEVERITIES and obv < OBV_SELL_STRONG:
        return (
            "blind_spot_review",
            "Modell sieht kein Risiko, aber Verkaufsdruck steigt — REVIEW empfohlen.",
        )

    return (
        "mixed",
        "Gemischte Signale — keine eindeutige Ausrichtung.",
    )
