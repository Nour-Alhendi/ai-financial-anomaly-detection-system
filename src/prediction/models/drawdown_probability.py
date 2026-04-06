"""
FinWatch AI — Drawdown Probability Model (v2)
=============================================

Question: "What is the probability of a drawdown > 5% in the next 20 days?"

Improvements in v2:
  - 15 new technical signal features (death_cross, RSI divergence, panic_volume, etc.)
  - Optuna hyperparameter tuning (--tune flag)
  - LightGBM comparison on every run (best model wins)

Output:
  p_drawdown   : float 0-1  — probability of >5% drawdown in 20 days
  drawdown_risk: "high" / "low"  (threshold: p_drawdown >= 0.45)
"""

import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Ensure `src/` is on the path when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prediction.features.technical_signals import add_vix, add_stock_ma_features, add_technical_signals

ROOT      = Path(__file__).resolve().parents[3]
DATA_DIR  = ROOT / "data/detection"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH      = MODEL_DIR / "xgboost_drawdown.pkl"
LGBM_MODEL_PATH = MODEL_DIR / "lgbm_drawdown.pkl"

HORIZON        = 10
THRESHOLD      = 0.05
TRAIN_DATA_END = pd.Timestamp("2026-03-01")

FEATURES = [
    # Anomaly signals
    'anomaly_score', 'anomaly_score_weighted',
    'z_score', 'z_score_60',
    'ae_error', 'ae_anomaly', 'if_anomaly',
    # Returns & volatility
    'returns', 'volatility', 'vol_5', 'vol_20', 'vol_change', 'volatility_ratio',
    # Momentum
    'momentum_5', 'momentum_10', 'trend_strength', 'rsi',
    'return_lag_1', 'return_lag_2', 'return_lag_3',
    # Drawdown history
    'max_drawdown_30d',
    # Volume
    'volume_zscore', 'is_high_volume', 'volume_trend',
    # OBV
    'obv_signal',
    # Market & sector context
    'spx_return', 'etf_return', 'relative_return', 'excess_return',
    'sector_relative', 'is_market_wide', 'is_sector_wide',
    'beta', 'regime_encoded',
    # Stock MA position
    'price_vs_ma200_stock', 'price_vs_ma50_stock',
    # Rolling stats
    'rolling_mean', 'rolling_std', 'rolling_std_60',
    # Tail risk
    'var_95', 'es_95', 'es_ratio',
    # VIX
    'vix_level', 'vix_change', 'vix_high',
    # ── New technical signals (v2) ──────────────────────────────────────────
    'price_vs_ema20',           # (Close / EMA20) - 1: short-term trend position
    'ma50_above_ma200',         # 1 = golden cross regime, 0 = death cross regime
    'golden_cross',             # MA50 just crossed above MA200 (last 20 days)
    'death_cross',              # MA50 just crossed below MA200 (last 20 days)
    'macd_cross_bullish',       # MACD crossed above signal line (last 3 days)
    'macd_cross_bearish',       # MACD crossed below signal line (last 3 days)
    'macd_hist',                # MACD histogram (continuous, sign = trend direction)
    'hh_hl',                    # Higher Highs + Higher Lows (uptrend structure)
    'll_lh',                    # Lower Lows + Lower Highs (downtrend structure)
    'panic_volume',             # Price drop > 2% + volume > 2x average
    'vol_ratio',                # Current volume / 20-day average volume
    'volume_spike_no_recovery', # Volume spike 2-5 days ago + still declining
    'rsi_divergence_bullish',   # Price lower lows but RSI higher lows
    'rsi_below_45_high_vol',    # RSI < 45 declining with rising volume
    'consolidation',            # Tight price range < 3% over 10 days
    'consolidation_above_ma50', # Consolidating above MA50 (bullish coil)
]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    dfs = []
    for f in sorted(DATA_DIR.glob("*.parquet")):
        if f.stem.startswith("^"):
            continue
        df = pd.read_parquet(f)
        df["ticker"] = f.stem
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values(["ticker", "Date"]).reset_index(drop=True)
    data = add_vix(data)
    data = add_stock_ma_features(data)
    data = add_technical_signals(data)
    return data


# ── Label generation ─────────────────────────────────────────────────────────

def generate_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Label = 1 if max drawdown in next HORIZON days exceeds THRESHOLD."""
    def _label_ticker(df):
        df = df.copy().sort_values("Date")
        closes = df["Close"].values
        labels = []
        for i in range(len(closes)):
            if i + HORIZON < len(closes):
                window = closes[i + 1: i + HORIZON + 1]
                dd = (window.min() - closes[i]) / closes[i]
                labels.append(1 if dd <= -THRESHOLD else 0)
            else:
                labels.append(np.nan)
        df["drawdown_event"] = labels
        return df

    data = pd.DataFrame(data.groupby("ticker", group_keys=False).apply(_label_ticker))
    return data.dropna(subset=["drawdown_event"])


def _prep_X(df: pd.DataFrame) -> pd.DataFrame:
    avail = [f for f in FEATURES if f in df.columns]
    return df[avail].apply(lambda c: pd.to_numeric(c, errors="coerce")).fillna(0)


# ── Optuna tuning ─────────────────────────────────────────────────────────────

def _tune_optuna(X_tr, y_tr, X_val, y_val, spw: float, n_trials: int = 80) -> dict:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  optuna not installed — pip install optuna")
        return {}

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 300, 1500),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel":trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
            "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 5.0),
        }
        model = XGBClassifier(
            **params,
            scale_pos_weight=spw,
            objective="binary:logistic",
            eval_metric="auc",
            early_stopping_rounds=30,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best Optuna AUC (val): {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params


# ── LightGBM comparison ───────────────────────────────────────────────────────

def _train_lgbm(X_tr, y_tr, X_val, y_val, X_test, y_test, spw: float) -> float:
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        print("  lightgbm not installed — pip install lightgbm  (skipping)")
        return 0.0

    model = LGBMClassifier(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.75,
        colsample_bytree=0.65,
        min_child_samples=20,
        reg_alpha=0.3,
        reg_lambda=3.0,
        scale_pos_weight=spw,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[],
    )

    probs = model.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, probs)
    print(f"  LightGBM Test AUC: {auc:.4f}")

    joblib.dump(model, LGBM_MODEL_PATH)
    return auc


# ── Training ──────────────────────────────────────────────────────────────────

def train(data: pd.DataFrame, tune: bool = False) -> XGBClassifier:
    train_mask = data["Date"] < pd.Timestamp("2024-01-01")
    test_mask  = (data["Date"] >= pd.Timestamp("2024-01-01")) & (data["Date"] < TRAIN_DATA_END)

    X_all = _prep_X(data)
    y_all = data["drawdown_event"].astype(int)

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

    split = int(len(X_train) * 0.9)
    X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
    y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]

    pos_rate = float(y_tr.mean())
    spw      = (1 - pos_rate) / pos_rate

    print(f"  Train rows: {len(X_tr):,}  |  Val rows: {len(X_val):,}  |  Test rows: {len(X_test):,}")
    print(f"  Positive rate: {pos_rate:.1%}  |  scale_pos_weight: {spw:.2f}")
    print(f"  Features used: {len([f for f in FEATURES if f in X_all.columns])} / {len(FEATURES)}")

    # ── Optuna tuning (optional) ───────────────────────────────────────────
    if tune:
        print("\n  Running Optuna hyperparameter search (80 trials)...")
        best_params = _tune_optuna(X_tr, y_tr, X_val, y_val, spw)
    else:
        best_params = {}

    xgb_params = {
        "n_estimators":      best_params.get("n_estimators",      1200),
        "max_depth":         best_params.get("max_depth",         5),
        "learning_rate":     best_params.get("learning_rate",     0.015),
        "subsample":         best_params.get("subsample",         0.75),
        "colsample_bytree":  best_params.get("colsample_bytree",  0.65),
        "colsample_bylevel": best_params.get("colsample_bylevel", 0.80),
        "min_child_weight":  best_params.get("min_child_weight",  15),
        "gamma":             best_params.get("gamma",             0.2),
        "reg_alpha":         best_params.get("reg_alpha",         0.3),
        "reg_lambda":        best_params.get("reg_lambda",        3.0),
    }

    # ── Train XGBoost ──────────────────────────────────────────────────────
    print("\n  Training XGBoost...")
    xgb_model = XGBClassifier(
        **xgb_params,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)

    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc   = roc_auc_score(y_test, xgb_probs)
    xgb_preds = (xgb_probs >= 0.45).astype(int)
    print(f"  XGBoost  — Best round: {xgb_model.best_iteration}  |  Test AUC: {xgb_auc:.4f}")
    print(classification_report(y_test, xgb_preds,
                                 target_names=["no_drawdown", "drawdown>5%"], digits=3))

    # ── Train LightGBM for comparison ─────────────────────────────────────
    print("\n  Training LightGBM for comparison...")
    lgbm_auc = _train_lgbm(X_tr, y_tr, X_val, y_val, X_test, y_test, spw)

    # ── Save best model ────────────────────────────────────────────────────
    if lgbm_auc > xgb_auc + 0.005:
        print(f"\n  LightGBM wins ({lgbm_auc:.4f} vs {xgb_auc:.4f})")
        best_path = LGBM_MODEL_PATH
    else:
        print(f"\n  XGBoost wins or ties ({xgb_auc:.4f} vs {lgbm_auc:.4f})")
        best_path = MODEL_PATH

    joblib.dump(xgb_model, MODEL_PATH)
    # Write a marker file so predict() knows which model to load
    (MODEL_DIR / "best_drawdown_model.txt").write_text(best_path.name)
    print(f"  → Best model: {best_path.name}  (saved marker)")
    return xgb_model


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(data: pd.DataFrame) -> pd.DataFrame:
    marker = MODEL_DIR / "best_drawdown_model.txt"
    best_path = MODEL_DIR / marker.read_text().strip() if marker.exists() else MODEL_PATH
    model  = joblib.load(best_path)
    latest = data.groupby("ticker").last().reset_index()
    X      = _prep_X(latest)
    avail  = [f for f in FEATURES if f in X.columns]
    probs  = model.predict_proba(X[avail])[:, 1]

    return pd.DataFrame({
        "ticker":        latest["ticker"].values,
        "p_drawdown":    probs.round(4),
        "drawdown_risk": ["high" if p >= 0.45 else "low" for p in probs],
    }).sort_values("p_drawdown", ascending=False).reset_index(drop=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def run():
    parser = argparse.ArgumentParser(description="Train drawdown probability model")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter search (80 trials, ~20 min)")
    args = parser.parse_args()

    print("Loading data...")
    data = load_data()
    print(f"Generating labels (>5% drawdown in {HORIZON} days)...")
    data = generate_labels(data)
    dist = data["drawdown_event"].value_counts(normalize=True)
    print(f"  Label distribution: drawdown={dist.get(1, 0):.1%}  no_drawdown={dist.get(0, 0):.1%}")

    print("\nTraining..." + (" (with Optuna tuning)" if args.tune else ""))
    train(data, tune=args.tune)

    print("\nPredictions (latest, top 15 by risk):")
    results = predict(data)
    print(results.head(15).to_string(index=False))


if __name__ == "__main__":
    run()
