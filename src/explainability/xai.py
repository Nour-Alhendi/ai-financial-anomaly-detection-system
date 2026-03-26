"""
FinWatch AI — Layer 7A: XAI (SHAP)
====================================
Loads XGBoost Risk model, computes SHAP values for the latest
row per ticker, and returns (latest_df, shap_matrix).

shap_matrix shape: (n_tickers, n_features)
Values: SHAP for the 'high' risk class
  positive → feature pushes model toward HIGH risk
  negative → feature pushes model toward LOW risk
"""

import joblib
import shap
import numpy as np
import pandas as pd
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"


def compute(data: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Returns (latest_df, shap_matrix).

    latest_df   — one row per ticker (most recent)
    shap_matrix — shape (n_tickers, n_features), SHAP for 'high' risk class
    """
    model = joblib.load(MODEL_DIR / "xgboost_risk.pkl")
    le    = joblib.load(MODEL_DIR / "risk_label_encoder.pkl")

    latest = data.groupby("ticker").last().reset_index()

    bool_cols = latest[features].select_dtypes(include="bool").columns
    latest[bool_cols] = latest[bool_cols].astype(int)
    X = (
        latest[features]
        .apply(lambda col: col.astype(float) if col.dtype.name == "category" else col)
        .fillna(0)
    )

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Binary XGBoost: shap_values can be 2D or 3D — always extract 'high' class
    classes = list(le.classes_)
    high_idx = next(
        (i for i, c in enumerate(classes) if str(c).lower() == "high"),
        len(classes) - 1,  # fallback: last class
    )
    if shap_values.ndim == 3:
        sv = shap_values[:, :, high_idx]
    else:
        sv = shap_values

    return latest, sv


# Features that describe market/regime context — used as background, not as the primary driver.
# Excluding these from driver selection forces the narrative to show the stock-specific signal.
CONTEXT_FEATURES = {"regime_encoded"}


def top3(shap_row: np.ndarray, features: list[str]) -> list[tuple[str, float]]:
    """
    Returns top-3 features by |SHAP| as [(feature_name, shap_val), ...].
    The first element is always the stock-specific driver (regime_encoded excluded).
    Elements 2-3 are the next highest by |SHAP| regardless of feature type.
    """
    all_pairs = sorted(
        zip(features, shap_row),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    # Driver: highest |SHAP| excluding context features
    driver = next(
        (p for p in all_pairs if p[0] not in CONTEXT_FEATURES),
        all_pairs[0],  # fallback: use top feature if all are context features
    )

    # Top-3 for display: driver first, then next 2 by |SHAP| (may include regime)
    rest = [p for p in all_pairs if p[0] != driver[0]][:2]
    return [driver] + rest


def format_top3(top3_list: list[tuple[str, float]]) -> str:
    parts = []
    for name, val in top3_list:
        sign = "+" if val >= 0 else ""
        parts.append(f"{name}({sign}{val:.3f})")
    return ", ".join(parts)
