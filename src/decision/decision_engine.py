"""
FinWatch AI — Layer 6: Decision Engine
=======================================
Core logic: Risk X Direction → Base Severity
Anomaly Score → Confidence only (not severity driver)
Hard Overrides → Drawdown threshold, REVIEW

Design Principles:
    1. Anomaly = signal (FLAG) to "look closer", not an action trigger
    2. Final decision = Risk X Direction
    3. Anomaly Score = confidence modifier only
    4. Every decision is auditable and traceable

Severity Levels:
    CRITICAL          — Immediate action required
    WARNING           — Elevated risk, monitor closely
    WATCH             — Low-level signal, observe
    NORMAL            — No significant signal
    POSITIVE_MOMENTUM — Strong upside with controlled risk
    REVIEW            — Conflicting signals, analyst check needed

Actions:
    ESCALATE  — Report to management immediately
    MONITOR   — Track closely, prepare for action
    OBSERVE   — Keep in watchlist
    NONE      — No action needed
    FLAG      — Manual analyst review required
"""

from dataclasses import dataclass
from typing import Optional


# ── Constants ─────────────

# Hard override thresholds
DRAWDOWN_CRITICAL   = -0.05     # -5%  → always CRITICAL regardless of other signals
DRAWDOWN_WARNING    = -0.03     # -3%  → at least WARNING if risk=high

# Relative risk threshold
# If market_anomaly=True AND stock's excess_return > this → market-wide move, not idiosyncratic
# Only stocks significantly underperforming the market warrant CRITICAL in a bear market
EXCESS_RETURN_CRITICAL = -0.02  # stock must underperform market by >2% to keep CRITICAL

# Direction confidence thresholds (from XGBoost)
P_DOWN_THRESHOLD    = 0.60      # minimum p_down to treat direction=down seriously
P_HIGH_THRESHOLD    = 0.60      # minimum p_high to treat risk=high seriously

# Positive momentum thresholds
P_DOWN_LOW          = 0.30      # p_down must be low for POSITIVE_MOMENTUM
RSI_OVERBOUGHT      = 70
RSI_OVERSOLD        = 30

# ES ratio threshold for tail risk
ES_RATIO_HIGH       = 1.5


# ── Data Classes ───────────

@dataclass
class AnomalyInput:
    """
    Structured input for one ticker/date decision.
    All fields come directly from Layer 4 + Layer 5 outputs.
    """
    ticker:         str
    date:           str

    # Risk Classifier (XGBoost)
    risk_level:     str           # "high" / "low"
    p_high:         float         # XGBoost confidence for risk=high (0-1)

    # Direction Classifier (XGBoost)
    direction:      str           # "up" / "stable" / "down"
    p_down:         float         # XGBoost confidence for direction=down (0-1)
    p_up:           float         # XGBoost confidence for direction=up (0-1)

    # Anomaly Detection (zscore, Isolation Forest, LSTM Autoencoder)
    anomaly_score:  int           # 0-4 (how many models flagged)
    market_anomaly: bool          # True if movement is market-wide
    sector_anomaly: bool          # True if movement is sector-wide

    # Expected Shortfall
    es_ratio:       float         # ES/VaR ratio — tail risk intensity

    # Features (for momentum signal)
    rsi:            float         # 0-100
    momentum_5:     float         # 5-day momentum
    momentum_10:    float         # 10-day momentum

    # Rolling max drawdown — from Layer 3 feature (max_drawdown_30d)
    drawdown:       float         # negative float, e.g. -0.04 = -4% (worst peak-to-trough, 30d)

    # Relative performance vs market (stock_return - market_return)
    excess_return:  float         # negative = underperforming market, positive = outperforming

    # OBV Signal — from Layer 5 feature (volume_zscore * returns)
    obv_signal:     float         # positive = buying pressure, negative = selling pressure


@dataclass
class DecisionOutput:
    """
    Auditable decision output for one ticker/date.
    Every field is traceable back to input signals.
    """
    ticker:           str
    date:             str
    severity:         str           # CRITICAL / WARNING / WATCH / NORMAL / POSITIVE_MOMENTUM / REVIEW
    action:           str           # ESCALATE / MONITOR / OBSERVE / NONE / FLAG
    confidence:       float         # anomaly_score / 4 — how confirmed is the signal (0-1)
    context:          str           # "idiosyncratic" / "market_wide" / "sector_wide"
    momentum_signal:  str           # "rising" / "falling" / "neutral"
    caution_flag:     Optional[str] # e.g. "Dead Cat Bounce possible"
    override_reason:  Optional[str] # set if a hard override was triggered
    summary:          str           # plain English for non-technical managers


# ── Helper Functions ───────────────────

def _confidence(anomaly_score: int) -> float:
    """
    Convert anomaly_score (0-4) to confidence ratio (0-1).
    Represents how many independent models confirmed the signal.
    """
    return round(anomaly_score / 4.0, 2)


def _context(market_anomaly: bool, sector_anomaly: bool) -> str:
    """
    Classify whether the signal is stock-specific, sector-wide, or market-wide.
    Used for report context — does NOT change severity.
    """
    if market_anomaly:
        return "market_wide"
    if sector_anomaly:
        return "sector_wide"
    return "idiosyncratic"


def _momentum_signal(rsi: float, momentum_5: float, momentum_10: float) -> str:
    """
    Classify momentum direction using RSI + momentum indicators.

    falling : RSI overbought AND short-term momentum weaker than mid-term
    rising  : both momentum values positive AND RSI not overbought
              OR RSI oversold (potential recovery)
    neutral : everything else
    """
    overbought  = rsi > RSI_OVERBOUGHT
    oversold    = rsi < RSI_OVERSOLD
    mom_falling = momentum_5 < momentum_10
    mom_rising  = momentum_5 > 0 and momentum_10 > 0

    if overbought and mom_falling:
        return "falling"
    if oversold:
        return "rising"
    if mom_rising and not overbought:
        return "rising"
    return "neutral"


def _summary(
    severity: str,
    context: str,
    caution_flag: Optional[str],
    override_reason: Optional[str],
    confidence: float
) -> str:
    """
    Generate plain English summary for non-technical managers.
    No model jargon — business language only.
    """
    context_note = {
        "market_wide":   "This movement is driven by broad market conditions, not the stock itself.",
        "sector_wide":   "This movement is shared across the sector, not isolated to this stock.",
        "idiosyncratic": "This signal is specific to this stock.",
    }.get(context, "")

    confidence_note = f"Signal confidence: {confidence:.0%}."

    base = {
        "CRITICAL":
            "Multiple risk indicators have triggered simultaneously. "
            "Immediate review is recommended.",
        "WARNING":
            "Elevated risk detected. Close monitoring is advised.",
        "WATCH":
            "A low-level signal has been detected. No immediate action needed, but worth observing.",
        "NORMAL":
            "No significant anomalies detected. Situation appears stable.",
        "POSITIVE_MOMENTUM":
            "Strong upward movement detected with controlled risk levels. "
            "Positive momentum confirmed by trend indicators.",
        "REVIEW":
            "Risk indicators are elevated but anomaly signals are absent or contradictory. "
            "Manual analyst review is recommended before any action.",
    }.get(severity, "Status unclear.")

    parts = [base, context_note, confidence_note]
    if caution_flag:
        parts.append(f"Note: {caution_flag}.")
    if override_reason:
        parts.append(f"Trigger: {override_reason}.")

    return " ".join(p for p in parts if p)


# ── Core Decision Logic ─────────────────

def decide(inp: AnomalyInput) -> DecisionOutput:
    """
    Main decision function.

    Priority order:
        1. Hard overrides      (drawdown — always fires regardless of other signals)
        2. REVIEW              (score=0 + risk=high — models contradict each other)
        3. ES Ratio Override   (tail risk too high, fires regardless of p_high confidence)
        4. Core Matrix         (risk=high + p_high >= threshold — main severity driver)
        5. Catch               (risk=high + p_high < threshold → WATCH, no silent fallthrough)
        6. Low Risk cases      (POSITIVE_MOMENTUM / WATCH)
        7. NORMAL              (default fallback)

    Anomaly score is NEVER the severity driver.
    It only affects the confidence value shown in the output.
    """
    confidence   = _confidence(inp.anomaly_score)
    context      = _context(inp.market_anomaly, inp.sector_anomaly)
    mom_signal   = _momentum_signal(inp.rsi, inp.momentum_5, inp.momentum_10)
    caution      = None
    override     = None

    # ── 1. Hard Override: Drawdown ────────────────────────────────────────────
    if inp.drawdown <= DRAWDOWN_CRITICAL:
        # Relative risk check: if stock is NOT significantly underperforming the market,
        # the drawdown is market-driven — downgrade to WARNING (not idiosyncratic)
        # Rule: market falls → WARNING; stock falls harder than market → CRITICAL
        if inp.excess_return > EXCESS_RETURN_CRITICAL:
            override = (
                f"Drawdown {inp.drawdown:.1%} but in line with market "
                f"(excess_return={inp.excess_return:+.1%})"
            )
            return DecisionOutput(
                ticker=inp.ticker, date=inp.date,
                severity="WARNING", action="MONITOR",
                confidence=confidence, context=context,
                momentum_signal=mom_signal,
                caution_flag=None, override_reason=override,
                summary=_summary("WARNING", context, None, override, confidence)
            )
        # Stock significantly underperforms market → CRITICAL (idiosyncratic risk)
        override = (
            f"Drawdown {inp.drawdown:.1%} AND underperforming market "
            f"by {inp.excess_return:+.1%}"
        )
        return DecisionOutput(
            ticker=inp.ticker, date=inp.date,
            severity="CRITICAL", action="ESCALATE",
            confidence=confidence, context=context,
            momentum_signal=mom_signal,
            caution_flag=None, override_reason=override,
            summary=_summary("CRITICAL", context, None, override, confidence)
        )

    if inp.drawdown <= DRAWDOWN_WARNING and inp.risk_level == "high":
        override = f"Drawdown {inp.drawdown:.1%} with high risk"
        return DecisionOutput(
            ticker=inp.ticker, date=inp.date,
            severity="WARNING", action="MONITOR",
            confidence=confidence, context=context,
            momentum_signal=mom_signal,
            caution_flag=None, override_reason=override,
            summary=_summary("WARNING", context, None, override, confidence)
        )

    # ── 2. REVIEW: Conflicting signals ────────────────────────────────────────
    # Risk model says high but no anomaly detected at all
    if inp.anomaly_score == 0 and inp.risk_level == "high" and inp.p_high >= P_HIGH_THRESHOLD:
        return DecisionOutput(
            ticker=inp.ticker, date=inp.date,
            severity="REVIEW", action="FLAG",
            confidence=confidence, context=context,
            momentum_signal=mom_signal,
            caution_flag="Risk model and anomaly detection contradict each other",
            override_reason=None,
            summary=_summary("REVIEW", context,
                             "Risk model and anomaly detection contradict each other",
                             None, confidence)
        )

    # ── 3. ES Ratio Override: Tail risk too high regardless of model confidence ─
    # Fires even if p_high < P_HIGH_THRESHOLD — ES captures tail danger independently
    if inp.risk_level == "high" and inp.es_ratio >= ES_RATIO_HIGH:
        return DecisionOutput(
            ticker=inp.ticker, date=inp.date,
            severity="WARNING", action="MONITOR",
            confidence=confidence, context=context,
            momentum_signal=mom_signal,
            caution_flag=f"Tail risk elevated: ES/VaR ratio = {inp.es_ratio:.2f}",
            override_reason=None,
            summary=_summary("WARNING", context,
                             f"Tail risk elevated: ES/VaR ratio = {inp.es_ratio:.2f}",
                             None, confidence)
        )

    # ── 4. Core Matrix: Risk x Direction ─────────────────────────────────────

    if inp.risk_level == "high" and inp.p_high >= P_HIGH_THRESHOLD:

        if inp.direction == "down" and inp.p_down >= P_DOWN_THRESHOLD:
            # OBV sanity check: strong buying pressure despite downward signal → downgrade
            if inp.obv_signal > 0.5:
                caution = "Institutional buying detected despite downward signal"
                return DecisionOutput(
                    ticker=inp.ticker, date=inp.date,
                    severity="WARNING", action="MONITOR",
                    confidence=confidence, context=context,
                    momentum_signal=mom_signal,
                    caution_flag=caution, override_reason=None,
                    summary=_summary("WARNING", context, caution, None, confidence)
                )
            # High risk + confirmed downward direction → CRITICAL
            return DecisionOutput(
                ticker=inp.ticker, date=inp.date,
                severity="CRITICAL", action="ESCALATE",
                confidence=confidence, context=context,
                momentum_signal=mom_signal,
                caution_flag=None, override_reason=None,
                summary=_summary("CRITICAL", context, None, None, confidence)
            )

        if inp.direction == "down" and inp.p_down < P_DOWN_THRESHOLD:
            # High risk but downward direction not confirmed → WARNING
            return DecisionOutput(
                ticker=inp.ticker, date=inp.date,
                severity="WARNING", action="MONITOR",
                confidence=confidence, context=context,
                momentum_signal=mom_signal,
                caution_flag="Downward direction signal is weak",
                override_reason=None,
                summary=_summary("WARNING", context,
                                 "Downward direction signal is weak", None, confidence)
            )

        if inp.direction == "stable":
            # High risk + no clear direction → WARNING
            return DecisionOutput(
                ticker=inp.ticker, date=inp.date,
                severity="WARNING", action="MONITOR",
                confidence=confidence, context=context,
                momentum_signal=mom_signal,
                caution_flag=None, override_reason=None,
                summary=_summary("WARNING", context, None, None, confidence)
            )

        if inp.direction == "up":
            # High risk + upward movement → ambiguous, flag Dead Cat Bounce
            caution = "Elevated risk despite upward movement — possible Dead Cat Bounce"
            return DecisionOutput(
                ticker=inp.ticker, date=inp.date,
                severity="WATCH", action="MONITOR",
                confidence=confidence, context=context,
                momentum_signal=mom_signal,
                caution_flag=caution, override_reason=None,
                summary=_summary("WATCH", context, caution, None, confidence)
            )

    # ── 5. Catch: risk=high but p_high below confidence threshold ────────────
    # Prevents silent fallthrough to NORMAL when model is uncertain but risk=high
    if inp.risk_level == "high" and inp.p_high < P_HIGH_THRESHOLD:
        return DecisionOutput(
            ticker=inp.ticker, date=inp.date,
            severity="WATCH", action="OBSERVE",
            confidence=confidence, context=context,
            momentum_signal=mom_signal,
            caution_flag="High risk signal with low model confidence",
            override_reason=None,
            summary=_summary("WATCH", context,
                             "High risk signal with low model confidence",
                             None, confidence)
        )

    # ── 6. Low Risk cases ─────────────────────────────────────────────────────

    if inp.risk_level == "low":

        # POSITIVE_MOMENTUM: confirmed upside + rising momentum + stock-specific
        if (
            inp.direction == "up"
            and inp.p_up >= P_DOWN_THRESHOLD
            and mom_signal == "rising"
            and inp.p_down < P_DOWN_LOW
            and context == "idiosyncratic"
        ):
            return DecisionOutput(
                ticker=inp.ticker, date=inp.date,
                severity="POSITIVE_MOMENTUM", action="OBSERVE",
                confidence=confidence, context=context,
                momentum_signal=mom_signal,
                caution_flag=None, override_reason=None,
                summary=_summary("POSITIVE_MOMENTUM", context, None, None, confidence)
            )

        if inp.direction == "down" and inp.p_down >= P_DOWN_THRESHOLD:
            # Low risk + confirmed downward direction → WATCH
            return DecisionOutput(
                ticker=inp.ticker, date=inp.date,
                severity="WATCH", action="OBSERVE",
                confidence=confidence, context=context,
                momentum_signal=mom_signal,
                caution_flag=None, override_reason=None,
                summary=_summary("WATCH", context, None, None, confidence)
            )

    # ── 7. NORMAL: Default fallback ───────────────────────────────────────────
    return DecisionOutput(
        ticker=inp.ticker, date=inp.date,
        severity="NORMAL", action="NONE",
        confidence=confidence, context=context,
        momentum_signal=mom_signal,
        caution_flag=None, override_reason=None,
        summary=_summary("NORMAL", context, None, None, confidence)
    )


# ── Batch Processing ──────────────────────────────────────────────────────────

def run_decision_engine(records: list[dict]) -> list[DecisionOutput]:
    """
    Process a list of ticker/date records through the decision engine.

    Args:
        records: list of dicts matching AnomalyInput fields

    Returns:
        list of DecisionOutput objects
    """
    results = []
    for r in records:
        inp = AnomalyInput(**r)
        results.append(decide(inp))
    return results


# ── Example Usage ───────────────────────────────
if __name__ == "__main__":
    examples = [
        {   # Expected: CRITICAL — drawdown override
            "ticker": "AAPL", "date": "2024-05-02",
            "risk_level": "high", "p_high": 0.88,
            "direction": "down", "p_down": 0.79, "p_up": 0.05,
            "anomaly_score": 4, "market_anomaly": False, "sector_anomaly": False,
            "es_ratio": 1.9, "rsi": 35.0,
            "momentum_5": -0.03, "momentum_10": -0.01, "drawdown": -0.06,
        },
        {   # Expected: CRITICAL — risk=high + direction=down confirmed
            "ticker": "MSFT", "date": "2024-05-02",
            "risk_level": "high", "p_high": 0.72,
            "direction": "down", "p_down": 0.68, "p_up": 0.10,
            "anomaly_score": 2, "market_anomaly": False, "sector_anomaly": False,
            "es_ratio": 1.6, "rsi": 42.0,
            "momentum_5": -0.02, "momentum_10": 0.01, "drawdown": -0.02,
        },
        {   # Expected: WATCH + Dead Cat Bounce — risk=high + direction=up
            "ticker": "TSLA", "date": "2024-05-02",
            "risk_level": "high", "p_high": 0.75,
            "direction": "up", "p_down": 0.15, "p_up": 0.65,
            "anomaly_score": 3, "market_anomaly": False, "sector_anomaly": False,
            "es_ratio": 1.7, "rsi": 55.0,
            "momentum_5": 0.02, "momentum_10": 0.01, "drawdown": -0.01,
        },
        {   # Expected: POSITIVE_MOMENTUM — low risk + confirmed upside
            "ticker": "NVDA", "date": "2024-05-02",
            "risk_level": "low", "p_high": 0.28,
            "direction": "up", "p_down": 0.08, "p_up": 0.75,
            "anomaly_score": 3, "market_anomaly": False, "sector_anomaly": False,
            "es_ratio": 0.9, "rsi": 62.0,
            "momentum_5": 0.04, "momentum_10": 0.02, "drawdown": -0.005,
        },
        {   # Expected: REVIEW — score=0 but risk=high
            "ticker": "JPM", "date": "2024-05-02",
            "risk_level": "high", "p_high": 0.80,
            "direction": "stable", "p_down": 0.30, "p_up": 0.25,
            "anomaly_score": 0, "market_anomaly": False, "sector_anomaly": False,
            "es_ratio": 1.7, "rsi": 50.0,
            "momentum_5": 0.0, "momentum_10": 0.0, "drawdown": -0.01,
        },
        {   # Expected: NORMAL — low risk, stable
            "ticker": "KO", "date": "2024-05-02",
            "risk_level": "low", "p_high": 0.22,
            "direction": "stable", "p_down": 0.20, "p_up": 0.30,
            "anomaly_score": 0, "market_anomaly": False, "sector_anomaly": False,
            "es_ratio": 0.8, "rsi": 48.0,
            "momentum_5": 0.001, "momentum_10": 0.002, "drawdown": -0.003,
        },
    ]

    decisions = run_decision_engine(examples)

    for d in decisions:
        print(f"\n{'='*65}")
        print(f"  {d.ticker} | {d.date}")
        print(f"  Severity   : {d.severity}")
        print(f"  Action     : {d.action}")
        print(f"  Confidence : {d.confidence:.0%}")
        print(f"  Context    : {d.context}")
        print(f"  Momentum   : {d.momentum_signal}")
        if d.caution_flag:
            print(f"  Caution    : {d.caution_flag}")
        if d.override_reason:
            print(f"  Override   : {d.override_reason}")
        print(f"  Summary    : {d.summary}")
